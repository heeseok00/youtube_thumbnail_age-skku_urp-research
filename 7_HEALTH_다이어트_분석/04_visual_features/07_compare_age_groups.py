"""
07_compare_age_groups.py
병합된 시각 피처를 34- vs 65+ 그룹 간 비교 분석 및 시각화.
(원본: 5_썸네일 피처 분석 파이프라인/analysis/compare_age_groups.py — 데이터 로딩만 변경.
 age_group은 샘플에 이미 부여되어 있으므로 재계산하지 않음)

입력: outputs/diet_visual_features.csv (06_merge_features.py 출력)
출력: outputs/analysis/*.png, *.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import OUT_DIR

INPUT_CSV = OUT_DIR / "diet_visual_features.csv"
OUTPUT_DIR = OUT_DIR / "analysis"

GROUP_YOUNG = "34-"
GROUP_OLD = "65+"


def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for f in candidates:
        if Path(f).exists():
            fm.fontManager.addfont(f)
            plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

CONTINUOUS_FEATURES = [
    ("person_ratio",       "인물 비중"),
    ("text_ratio",         "텍스트 비중"),
    ("color_saturation",   "색상 채도"),
    ("color_brightness",   "색상 밝기"),
    ("color_warm_ratio",   "따뜻한 색 비율"),
    ("color_entropy",      "색상 다양성(엔트로피)"),
    ("color_hue_std",      "색조 표준편차"),
]

CATEGORICAL_FEATURES = [
    ("person_count_cat",      "인물 수"),
    ("head_pose",             "인물 시선(헤드포즈)"),
    ("expression_dominant",   "인물 표정"),
]

TEXT_REGION_FEATURES = [
    ("text_color_entropy",       "텍스트 ROI 색 다양성"),
    ("text_color_saturation",    "텍스트 ROI 채도"),
    ("text_color_brightness",    "텍스트 ROI 밝기"),
    ("text_color_hue_std",       "텍스트 ROI 색조 std"),
    ("bg_color_entropy",         "배경 ROI 색 다양성"),
    ("text_bg_entropy_diff",     "텍스트-배경 entropy 차"),
    ("text_bg_saturation_diff",  "텍스트-배경 채도 차"),
]


def compare_continuous(df, feat, label, ax):
    """연속형 피처: Violin plot + Mann-Whitney U test."""
    g0 = df[df["age_group"] == GROUP_YOUNG][feat].dropna()
    g1 = df[df["age_group"] == GROUP_OLD][feat].dropna()

    if len(g0) < 5 or len(g1) < 5:
        ax.set_title(f"{label}\n(데이터 부족)")
        return None

    stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")

    plot_df = pd.concat([
        pd.DataFrame({"값": g0, "그룹": "~34세"}),
        pd.DataFrame({"값": g1, "그룹": "65세+"}),
    ])
    sns.violinplot(data=plot_df, x="그룹", y="값", hue="그룹", ax=ax,
                   palette=["#4C9BE8", "#E87B4C"], legend=False)
    sns.stripplot(data=plot_df, x="그룹", y="값", ax=ax, size=2, alpha=0.3, color="black")

    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.set_title(f"{label}\np={p:.4f} {sig}\n(~34세 n={len(g0)}, 65세+ n={len(g1)})", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return {"feature": feat, "label": label, "n_young": len(g0), "n_old": len(g1),
            "mean_young": round(g0.mean(), 4), "mean_old": round(g1.mean(), 4),
            "p_value": round(p, 6), "significance": sig}


def compare_categorical(df, feat, label, ax):
    """범주형 피처: Bar chart + Chi-square test."""
    sub = df[["age_group", feat]].dropna()
    if len(sub) < 10:
        ax.set_title(f"{label}\n(데이터 부족)")
        return None

    ct = pd.crosstab(sub[feat], sub["age_group"], normalize="columns") * 100
    ct.plot(kind="bar", stacked=False, ax=ax,
            color=["#4C9BE8", "#E87B4C"], width=0.6)

    chi2_table = pd.crosstab(sub[feat], sub["age_group"])
    chi2, p, dof, _ = stats.chi2_contingency(chi2_table)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    ax.set_title(f"{label}\nχ²={chi2:.2f}, p={p:.4f} {sig}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("비율 (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7)
    return {"feature": feat, "label": label, "chi2": round(chi2, 4),
            "p_value": round(p, 6), "dof": dof, "significance": sig}


def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input or INPUT_CSV, low_memory=False)
    cnt = df["age_group"].value_counts()
    print(f"[대상] 다이어트 샘플 {len(df):,}개 "
          f"(34-: {cnt.get(GROUP_YOUNG, 0):,} / 65+: {cnt.get(GROUP_OLD, 0):,})")

    # ── 연속형 피처 ──────────────────────────────────────────────────────────
    avail_cont = [(f, l) for f, l in CONTINUOUS_FEATURES if f in df.columns]
    if avail_cont:
        fig, axes = plt.subplots(1, len(avail_cont), figsize=(4 * len(avail_cont), 5))
        if len(avail_cont) == 1:
            axes = [axes]
        fig.suptitle("연속형 피처 비교: ~34세 vs 65세+ (다이어트)", fontsize=12, y=1.01)
        cont_results = []
        for ax, (feat, label) in zip(axes, avail_cont):
            result = compare_continuous(df, feat, label, ax)
            if result:
                cont_results.append(result)
        plt.tight_layout()
        cont_path = OUTPUT_DIR / "01_continuous_features.png"
        plt.savefig(cont_path, dpi=150, bbox_inches="tight")
        print(f"[저장] {cont_path}")
        plt.close()
        if cont_results:
            pd.DataFrame(cont_results).to_csv(OUTPUT_DIR / "01_continuous_stats.csv", index=False)

    # ── 범주형 피처 ──────────────────────────────────────────────────────────
    avail_cat = [(f, l) for f, l in CATEGORICAL_FEATURES if f in df.columns]
    if avail_cat:
        fig, axes = plt.subplots(1, len(avail_cat), figsize=(5 * len(avail_cat), 5))
        if len(avail_cat) == 1:
            axes = [axes]
        fig.suptitle("범주형 피처 비교: ~34세 vs 65세+ (다이어트)", fontsize=12, y=1.01)
        cat_results = []
        for ax, (feat, label) in zip(axes, avail_cat):
            result = compare_categorical(df, feat, label, ax)
            if result:
                cat_results.append(result)
        plt.tight_layout()
        cat_path = OUTPUT_DIR / "02_categorical_features.png"
        plt.savefig(cat_path, dpi=150, bbox_inches="tight")
        print(f"[저장] {cat_path}")
        plt.close()
        if cat_results:
            pd.DataFrame(cat_results).to_csv(OUTPUT_DIR / "02_categorical_stats.csv", index=False)

    # ── 텍스트 ROI 색상 (text_ratio > 0.05) ─────────────────────────────────
    avail_text = [(f, l) for f, l in TEXT_REGION_FEATURES if f in df.columns]
    if avail_text and "text_ratio" in df.columns:
        text_sub = df[(df["text_ratio"] > 0.05) & df["text_color_entropy"].notna()].copy()
        print(f"[텍스트 ROI 분석] text_ratio>0.05 & entropy 유효: {len(text_sub):,}개")
        if len(text_sub) >= 20:
            ncols = min(len(avail_text), 4)
            nrows = (len(avail_text) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
            axes = np.array(axes).reshape(-1)
            fig.suptitle("텍스트 ROI 색상 (text_ratio>0.05): ~34세 vs 65세+ (다이어트)", fontsize=12, y=1.01)
            text_results = []
            for ax, (feat, label) in zip(axes, avail_text):
                result = compare_continuous(text_sub, feat, label, ax)
                if result:
                    text_results.append(result)
            for ax in axes[len(avail_text):]:
                ax.set_visible(False)
            plt.tight_layout()
            text_path = OUTPUT_DIR / "03_text_region_color.png"
            plt.savefig(text_path, dpi=150, bbox_inches="tight")
            print(f"[저장] {text_path}")
            plt.close()
            if text_results:
                pd.DataFrame(text_results).to_csv(
                    OUTPUT_DIR / "03_text_region_color_stats.csv", index=False
                )

    print(f"\n[완료] 결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 피처 연령대 비교 분석")
    parser.add_argument("--input", default=None)
    main(parser.parse_args())
