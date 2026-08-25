"""Step 2 — title 피처 기반 34- vs 65+ 분류 + SHAP 변수 중요도 + 그룹 통계.

정은의 title_features_shap.py를 다이어트 표본에 맞게 튜닝.
- 입력: outputs/diet_title_features.csv (Step 1 출력, target 컬럼 포함)
- 라벨: 연령 비율 재계산 대신 표본의 target 사용
- RF/XGBoost/LightGBM 5-fold CV 비교 → 최고 모델 SHAP → Mann-Whitney U 검정 (원본 그대로)
- 출력: outputs/figures/shap_summary_*.png + 콘솔 표

실행: python 02_title_features_shap.py
"""
import warnings
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

STAGE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = STAGE_DIR / "outputs/diet_title_features.csv"
FIG_DIR = STAGE_DIR / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "char_count", "word_count", "mean_word_length", "ttr",
    "exclamation_count", "ellipsis_count", "quotation_count",
    "bracket_count", "common_punct_count",
    "numbers_count", "emoji_count",
    "has_list_number", "personal_pronoun_count",
    "superlative_count", "curiosity_count",
    "noun_ratio", "verb_ratio", "adv_ratio", "nnp_ratio",
    "has_pronoun", "has_past_tense", "has_question",
    "is_sentence", "is_noun_phrase",
    "clip_similarity",
]


def load_data(path=FEATURES_PATH):
    df = pd.read_csv(path)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    df_bin = df.dropna(subset=feature_cols + ["target"]).copy()
    df_bin["target"] = df_bin["target"].astype(int)

    X = df_bin[feature_cols].values
    y = df_bin["target"].values
    print(f"샘플 수: {len(y)}개 | 34-: {(y == 0).sum()}개 | 65+: {(y == 1).sum()}개")
    return X, y, feature_cols


def train_and_compare(X, y):
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, average="binary"),
        "roc_auc": "roc_auc",
    }
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1, verbose=-1,
            num_leaves=31, min_child_samples=20,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        print(f"\n[{name}] 학습 중...")
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "accuracy": scores["test_accuracy"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
        }
        print(f"  Accuracy : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
        print(f"  F1       : {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")
        print(f"  ROC-AUC  : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")

    df_summary = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in results.items()]
    ).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 60)
    print("전체 요약 (ROC-AUC 기준)")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    return models, df_summary


def run_shap(models, df_summary, X, y, feature_cols, out_dir=FIG_DIR):
    best_model_name = df_summary.iloc[0]["model"]
    print(f"\n[SHAP] 사용 모델: {best_model_name}")

    model = models[best_model_name]
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(X, columns=feature_cols))

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_neg = shap_values[:, :, 0]
        shap_pos = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        shap_neg, shap_pos = shap_values[0], shap_values[1]
    else:
        shap_pos, shap_neg = shap_values, -shap_values

    X_df = pd.DataFrame(X, columns=feature_cols)

    for label, values in [("65+", shap_pos), ("~34", shap_neg)]:
        shap.summary_plot(values, X_df, plot_type="dot", show=False, max_display=10, plot_size=(14, 10))
        plt.title(f"{label} Group SHAP Summary", fontsize=28, fontweight="bold")
        plt.xlabel("SHAP value (impact on model output)", fontsize=20, fontweight="bold")
        plt.ylabel("Feature", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{label.replace('+', 'plus')}.png", dpi=150)
        plt.close()

    mean_shap_pos = pd.Series(shap_pos.mean(axis=0), index=feature_cols)
    mean_shap_neg = pd.Series(shap_neg.mean(axis=0), index=feature_cols)

    print("\n[65+ 기준 평균 SHAP Top 10] (양수=65+ 예측에 기여)")
    print(mean_shap_pos.sort_values(ascending=False).head(10).round(4).to_string())

    print("\n[34- 기준 평균 SHAP Top 10] (양수=34- 예측에 기여)")
    print(mean_shap_neg.sort_values(ascending=False).head(10).round(4).to_string())

    shap_comparison = pd.DataFrame({
        "feature": feature_cols,
        "mean_shap_65+": shap_pos.mean(axis=0).round(4),
        "mean_shap_~34": shap_neg.mean(axis=0).round(4),
    }).sort_values("mean_shap_65+", ascending=False)
    print("\n[전체 피처별 평균 SHAP 비교]")
    print(shap_comparison.to_string(index=False))
    shap_comparison.to_csv(out_dir / "shap_feature_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"\n[저장] {out_dir}/shap_summary_65plus.png / shap_summary_~34.png / shap_feature_comparison.csv")
    return X_df


def group_stats(X_df, y, feature_cols, out_dir=FIG_DIR):
    X_df = X_df.copy()
    X_df["target"] = y

    group_mean = X_df.groupby("target")[feature_cols].mean().T
    group_mean.columns = ["34-", "65+"]
    print("\n[그룹별 피처 평균]")
    print(group_mean.round(4).to_string())

    rows = []
    for col in feature_cols:
        group_34 = X_df[X_df["target"] == 0][col]
        group_65 = X_df[X_df["target"] == 1][col]
        stat, p = mannwhitneyu(group_34, group_65, alternative="two-sided")
        rows.append({
            "feature": col,
            "mean_~34": round(group_34.mean(), 4),
            "mean_65+": round(group_65.mean(), 4),
            "p_value": round(p, 6),
        })
    df_mw = pd.DataFrame(rows).sort_values("p_value")
    print("\n[Mann-Whitney U test]")
    print(df_mw.to_string(index=False))
    df_mw.to_csv(out_dir / "mannwhitney_results.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    X, y, feature_cols = load_data()
    models, df_summary = train_and_compare(X, y)
    X_df = run_shap(models, df_summary, X, y, feature_cols)
    group_stats(X_df, y, feature_cols)
