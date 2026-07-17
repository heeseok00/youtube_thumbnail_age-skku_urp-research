"""
multivariate_analysis.py
피처 간 상관관계 분석 + Random Forest 피처 중요도 분석

출력 (outputs/per_feature/):
    corr_matrix.png          - 피처 간 Spearman 상관계수 히트맵
    rf_feature_importance.png - Random Forest 피처 중요도 바 차트
    rf_feature_importance.csv - 수치 데이터
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs/per_feature"

MERGED_CSV = BASE_DIR / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title_features_merged.csv"

AGE_COLS   = ["age_~17","age_18~24","age_25~34","age_35~44","age_45~54","age_55~64","age_65~"]
YOUNG_COLS = ["age_~17","age_18~24","age_25~34"]

# 분석에 사용할 피처 (연속형만 — RF에 직접 투입)
CONT_FEATURES = [
    ("person_ratio",     "인물 비중"),
    ("text_ratio",       "텍스트 비중"),
    ("color_saturation", "색상 채도"),
    ("color_brightness", "색상 밝기"),
    ("color_warm_ratio", "따뜻한 색 비율"),
    ("color_entropy",    "색상 다양성"),
    ("color_hue_std",    "색조 표준편차"),
]

# 범주형 → 수치 인코딩 피처
CAT_FEATURES = [
    ("person_count",      "인물 수"),
    ("head_pose_front",   "정면 응시"),         # head_pose == 정면 → 1/0
    ("expression_sad",    "슬픔 확률"),
    ("expression_happy",  "기쁨 확률"),
    ("expression_neutral","중립 확률"),
]


def set_korean_font():
    for f in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(f).exists():
            fm.fontManager.addfont(f)
            plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def load_data():
    import matplotlib
    matplotlib.use("Agg")

    df = pd.read_csv(MERGED_CSV, low_memory=False)
    df = df[df["subcategory"] == "시사/뉴스/사건"].copy().reset_index(drop=True)

    def assign_group(row):
        d = row[AGE_COLS].idxmax()
        if d in YOUNG_COLS: return 0   # ~34세
        if d == "age_65~":  return 1   # 65세+
        return None
    df["label"] = df.apply(assign_group, axis=1)
    df = df[df["label"].notna()].copy()

    # 범주형 피처 수치화
    df["head_pose_front"] = (df["head_pose"] == "정면").astype(float)

    return df


def plot_correlation(df, feat_cols, feat_labels):
    """Spearman 상관계수 히트맵."""
    data = df[feat_cols].dropna()
    corr = data.corr(method="spearman")
    corr.index   = feat_labels
    corr.columns = feat_labels

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # 상삼각 마스크
    sns.heatmap(corr, ax=ax, mask=mask,
                annot=True, fmt=".2f", annot_kws={"size": 9},
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("피처 간 Spearman 상관계수\n(시사/뉴스/사건, ~34세+65세+ 영상 전체)", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    out = OUTPUT_DIR / "corr_matrix.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out}")


def plot_rf_importance(df, feat_cols, feat_labels):
    """Random Forest 피처 중요도."""
    sub = df[feat_cols + ["label"]].dropna()
    X = sub[feat_cols].values
    y = sub["label"].values.astype(int)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                random_state=42, n_jobs=-1)

    # 5-fold CV 정확도
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"[RF] 5-fold CV 정확도: {scores.mean():.3f} ± {scores.std():.3f}")

    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)

    colors = ["#E87B4C" if importances[i] >= np.median(importances) else "#B0BEC5"
              for i in order]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.barh([feat_labels[i] for i in order],
                   [importances[i] for i in order],
                   color=colors, height=0.6)
    ax.axvline(np.median(importances), color="gray", linewidth=1,
               linestyle="--", alpha=0.7, label="중앙값")

    for bar, i in zip(bars, order):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{importances[i]:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(
        f"Random Forest 피처 중요도\n(~34세 vs 65세+, 5-fold CV 정확도={scores.mean():.3f})",
        fontsize=11
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "rf_feature_importance.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out}")

    result_df = pd.DataFrame({
        "feature": feat_cols,
        "label":   feat_labels,
        "importance": importances
    }).sort_values("importance", ascending=False)
    result_df.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)
    print(result_df.to_string(index=False))
    return result_df


def main():
    import matplotlib
    matplotlib.use("Agg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    print("[로드] 데이터 준비 중...")
    df = load_data()
    print(f"  ~34세: {(df['label']==0).sum():,}  /  65세+: {(df['label']==1).sum():,}")

    all_feat_cols   = [f for f, _ in CONT_FEATURES] + [f for f, _ in CAT_FEATURES]
    all_feat_labels = [l for _, l in CONT_FEATURES] + [l for _, l in CAT_FEATURES]

    # 상관관계 분석: 연속형 피처만
    cont_cols   = [f for f, _ in CONT_FEATURES]
    cont_labels = [l for _, l in CONT_FEATURES]
    print("\n[1] 피처 간 상관관계 분석...")
    plot_correlation(df, cont_cols, cont_labels)

    # Random Forest: 전체 피처
    print("\n[2] Random Forest 피처 중요도 분석...")
    plot_rf_importance(df, all_feat_cols, all_feat_labels)

    print(f"\n[완료] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
