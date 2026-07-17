"""
visualize_per_feature.py
각 피처를 개별 파일로 시각화 + 각 그룹의 실제 썸네일 샘플 비교

출력 (outputs/per_feature/):
    feat_<name>.png          - 통계 시각화 (violin / bar)
    samples_<name>.png       - 두 그룹 썸네일 샘플 4장씩 비교
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

BASE_DIR   = Path(__file__).parent.parent.parent          # 26-1_URP/
OUTPUT_DIR = Path(__file__).parent / "outputs/per_feature"

MERGED_CSV = BASE_DIR / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title_features_merged.csv"
SRC_CSV    = BASE_DIR / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
THUMB_ROOT = BASE_DIR                                     # thumbnail_path 앞에 붙일 루트

AGE_COLS   = ["age_~17", "age_18~24", "age_25~34", "age_35~44", "age_45~54", "age_55~64", "age_65~"]
YOUNG_COLS = ["age_~17", "age_18~24", "age_25~34"]

CONTINUOUS_FEATURES = [
    ("person_ratio",     "인물 비중"),
    ("text_ratio",       "텍스트 비중"),
    ("color_saturation", "색상 채도"),
    ("color_brightness", "색상 밝기"),
    ("color_warm_ratio", "따뜻한 색 비율"),
    ("color_entropy",    "색상 다양성(엔트로피)"),
    ("color_hue_std",    "색조 표준편차"),
]

CATEGORICAL_FEATURES = [
    ("person_count_cat",    "인물 수"),
    ("head_pose",           "인물 시선(헤드포즈)"),
    ("expression_dominant", "인물 표정"),
]

# 사람이 있는 영상(person_ratio > 0)만 분석할 피처 목록
PERSON_ONLY_FEATURES = {"person_ratio", "head_pose", "expression_dominant"}

COLORS = {"~34세": "#4C9BE8", "65세+": "#E87B4C"}

# ── 폰트 ──────────────────────────────────────────────────────────────────────
def set_korean_font():
    for f in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(f).exists():
            fm.fontManager.addfont(f)
            plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
            return
    plt.rcParams["font.family"] = "DejaVu Sans"

set_korean_font()
plt.rcParams["axes.unicode_minus"] = False


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
def load_data():
    merged = pd.read_csv(MERGED_CSV, low_memory=False)
    src    = pd.read_csv(SRC_CSV, low_memory=False, usecols=["video_id", "thumbnail_path"])
    df = merged.merge(src, on="video_id", how="left")
    df = df[df["subcategory"] == "시사/뉴스/사건"].copy().reset_index(drop=True)

    def assign_group(row):
        dominant = row[AGE_COLS].idxmax()
        if dominant in YOUNG_COLS:  return "~34세"
        if dominant == "age_65~":   return "65세+"
        return None

    df["age_group"] = df.apply(assign_group, axis=1)
    df["thumb_abs"] = df["thumbnail_path"].str.replace("\\", "/", regex=False).apply(
        lambda p: str(THUMB_ROOT / p) if pd.notna(p) else None
    )
    return df


def resolve_img(path, size=(160, 90)):
    """썸네일 이미지를 읽어 RGB numpy 배열로 반환. 실패 시 회색 placeholder."""
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return img
    except Exception:
        return np.full((*size[::-1], 3), 200, dtype=np.uint8)


# ── 통계 시각화 ───────────────────────────────────────────────────────────────
def plot_continuous(df_sub, feat, label, person_only=False):
    g0 = df_sub[df_sub["age_group"] == "~34세"][feat].dropna()
    g1 = df_sub[df_sub["age_group"] == "65세+"][feat].dropna()
    _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    plot_df = pd.concat([
        pd.DataFrame({"값": g0, "그룹": "~34세"}),
        pd.DataFrame({"값": g1, "그룹": "65세+"}),
    ])

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(data=plot_df, x="그룹", y="값", hue="그룹", ax=ax,
                   palette=COLORS, legend=False)
    sns.stripplot(data=plot_df, x="그룹", y="값", ax=ax,
                  size=2, alpha=0.25, color="black", jitter=True)

    ax.axhline(g0.mean(), color=COLORS["~34세"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(g1.mean(), color=COLORS["65세+"],  linestyle="--", linewidth=1.2, alpha=0.8)

    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    filter_note = "\n(사람 있는 영상만, person_ratio > 0)" if person_only else ""
    ax.set_title(
        f"{label}{filter_note}\n"
        f"~34세 평균={g0.mean():.4f} (n={len(g0):,})  /  65세+ 평균={g1.mean():.4f} (n={len(g1):,})\n"
        f"Mann-Whitney U  p={p_str} {sig}",
        fontsize=9
    )
    ax.set_xlabel("")
    ax.set_ylabel(label)
    plt.tight_layout()
    return fig


def plot_categorical(df_sub, feat, label, person_only=False):
    sub = df_sub[df_sub["age_group"].isin(["~34세", "65세+"])][["age_group", feat]].dropna()
    ct  = pd.crosstab(sub[feat], sub["age_group"], normalize="columns") * 100
    chi2_tbl = pd.crosstab(sub[feat], sub["age_group"])
    chi2, p, dof, _ = stats.chi2_contingency(chi2_tbl)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    n_young = (df_sub["age_group"] == "~34세").sum()
    n_old   = (df_sub["age_group"] == "65세+").sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(ct.index))
    w = 0.35
    ax.bar(x - w/2, ct.get("~34세", 0), width=w, color=COLORS["~34세"], label=f"~34세 (n={n_young:,})")
    ax.bar(x + w/2, ct.get("65세+",  0), width=w, color=COLORS["65세+"],  label=f"65세+ (n={n_old:,})")
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=30, ha="right")
    ax.set_ylabel("비율 (%)")
    ax.legend()
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    filter_note = "\n(사람 있는 영상만, person_ratio > 0)" if person_only else ""
    ax.set_title(
        f"{label}{filter_note}\n"
        f"χ²={chi2:.2f}, p={p_str} {sig}  (dof={dof})",
        fontsize=10
    )
    plt.tight_layout()
    return fig


# ── 썸네일 샘플 시각화 ────────────────────────────────────────────────────────
def pick_samples(df_sub, feat, group, n=4, mode="high"):
    """feat 값이 높은(또는 낮은) 순 n개 샘플 선택."""
    g = df_sub[df_sub["age_group"] == group].dropna(subset=[feat, "thumb_abs"]).copy()
    if pd.api.types.is_numeric_dtype(g[feat]):
        g = g.sort_values(feat, ascending=(mode == "low"))
    else:
        # 범주형: 최빈값 카테고리 우선
        top_cat = g[feat].value_counts().index[0]
        g = g[g[feat] == top_cat]
    return g.head(n)


def plot_samples(df_sub, feat, label, n=4):
    """두 그룹에서 각각 n장 썸네일 샘플 비교 이미지 생성."""
    groups   = ["~34세", "65세+"]
    fig, axes = plt.subplots(2, n, figsize=(n * 2.8, 6))
    fig.suptitle(f"[{label}] 그룹별 썸네일 샘플", fontsize=12, y=1.01)

    for row_i, grp in enumerate(groups):
        samples = pick_samples(df_sub, feat, grp, n=n, mode="high")
        for col_i in range(n):
            ax = axes[row_i][col_i]
            ax.axis("off")
            if col_i < len(samples):
                row_data = samples.iloc[col_i]
                img = resolve_img(row_data["thumb_abs"])
                ax.imshow(img)
                val = row_data[feat]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                ax.set_title(f"{val_str}", fontsize=7)
            if col_i == 0:
                ax.set_ylabel(grp, fontsize=10, rotation=0,
                              labelpad=55, va="center",
                              color=COLORS[grp], fontweight="bold")
    plt.tight_layout()
    return fig


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[로드] 데이터 준비 중...")
    df = load_data()
    sub = df[df["age_group"].isin(["~34세", "65세+"])].copy()
    print(f"  ~34세: {(sub['age_group']=='~34세').sum():,}  /  65세+: {(sub['age_group']=='65세+').sum():,}")

    # person_ratio > 0 필터 적용한 서브셋
    sub_with_person = sub[sub["person_ratio"] > 0].copy()

    # ── 연속형 ──
    for feat, label in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            print(f"[스킵] {feat} 컬럼 없음")
            continue

        data = sub_with_person if feat in PERSON_ONLY_FEATURES else sub
        filter_note = " [사람 있는 영상만]" if feat in PERSON_ONLY_FEATURES else ""
        print(f"[처리] {label} ({feat}){filter_note}  n={len(data):,}")

        fig = plot_continuous(data, feat, label, person_only=(feat in PERSON_ONLY_FEATURES))
        fig.savefig(OUTPUT_DIR / f"feat_{feat}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_samples(data, feat, label)
        fig.savefig(OUTPUT_DIR / f"samples_{feat}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 범주형 ──
    for feat, label in CATEGORICAL_FEATURES:
        if feat not in df.columns:
            print(f"[스킵] {feat} 컬럼 없음")
            continue

        data = sub_with_person if feat in PERSON_ONLY_FEATURES else sub
        filter_note = " [사람 있는 영상만]" if feat in PERSON_ONLY_FEATURES else ""
        print(f"[처리] {label} ({feat}){filter_note}  n={len(data):,}")

        fig = plot_categorical(data, feat, label, person_only=(feat in PERSON_ONLY_FEATURES))
        fig.savefig(OUTPUT_DIR / f"feat_{feat}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_samples(data, feat, label)
        fig.savefig(OUTPUT_DIR / f"samples_{feat}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[완료] 저장 위치: {OUTPUT_DIR}")
    print(f"  생성된 파일 수: {len(list(OUTPUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()


def plot_person_presence():
    """사람 등장 여부 비율 비교 + 카이제곱 검정."""
    import matplotlib.patches as mpatches
    from scipy.stats import chi2_contingency

    data = {
        "~34세": {"사람 있음": 2197, "사람 없음": 598},
        "65세+": {"사람 있음": 2592, "사람 없음": 490},
    }

    groups  = list(data.keys())
    totals  = {g: sum(data[g].values()) for g in groups}
    pct_no  = {g: data[g]["사람 없음"] / totals[g] * 100 for g in groups}
    pct_yes = {g: data[g]["사람 있음"] / totals[g] * 100 for g in groups}

    # 카이제곱 검정
    ct = [[data[g]["사람 있음"], data[g]["사람 없음"]] for g in groups]
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    fig, ax = plt.subplots(figsize=(5, 5))
    x = [0, 1]
    w = 0.5
    bars_yes = ax.bar(x, [pct_yes[g] for g in groups], width=w,
                      color=[COLORS[g] for g in groups], alpha=0.85, label="사람 있음")
    bars_no  = ax.bar(x, [pct_no[g] for g in groups],  width=w,
                      bottom=[pct_yes[g] for g in groups],
                      color=[COLORS[g] for g in groups], alpha=0.4, hatch="//", label="사람 없음")

    # 각 구간 비율 텍스트
    for i, g in enumerate(groups):
        ax.text(i, pct_yes[g] / 2,       f"{pct_yes[g]:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(i, pct_yes[g] + pct_no[g] / 2, f"{pct_no[g]:.1f}%",  ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={totals[g]:,})" for g in groups], fontsize=11)
    ax.set_ylabel("비율 (%)")
    ax.set_ylim(0, 115)
    ax.set_title(
        f"썸네일 내 사람 등장 여부\n"
        f"χ²={chi2:.2f}, p={p_str} {sig}",
        fontsize=11
    )

    yes_patch = mpatches.Patch(color="gray", alpha=0.85, label="사람 있음 (person_ratio > 0)")
    no_patch  = mpatches.Patch(color="gray", alpha=0.4,  hatch="//", label="사람 없음 (person_ratio = 0)")
    ax.legend(handles=[yes_patch, no_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "feat_person_presence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {out}")


if __name__ == "__main__":
    set_korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_person_presence()
