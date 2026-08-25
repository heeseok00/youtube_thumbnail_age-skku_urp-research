"""Step 3 — 통계 검증 (독립 실행 가능하도록 튜닝).

1) 클러스터별 연령대 키워드 편향: Fisher's Exact Test
   - 원본과 달리 전체 단어 수(100/100 하드코딩)를 데이터에서 자동 계산
2) XGBoost 분류 성능: 다수 클래스 베이스라인 대비 단측 이항검정
   - 원본은 Step 2 세션 메모리(model, X_test)에 의존했으나,
     Step 2가 저장한 outputs/figures/test_predictions.csv를 읽어 독립 실행

실행: python 03_stats_validation.py (Step 2 완료 후)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import binomtest

FIG_DIR = Path(__file__).resolve().parent / "outputs/figures"

# ── 1. 클러스터 편향 검정 (Fisher's Exact) ────────────────────────────────────
df = pd.read_csv(FIG_DIR / "table3_cluster_summary.csv")

col_65, col_34 = "n_65plus_keywords", "n_under34_keywords"
total_65 = int(df[col_65].sum())
total_34 = int(df[col_34].sum())
print(f"전체 키워드: 65+ {total_65}개 / ~34 {total_34}개")

p_values, significance = [], []
for _, row in df.iterrows():
    c65, c34 = int(row[col_65]), int(row[col_34])
    table = [[c65, c34], [total_65 - c65, total_34 - c34]]
    p = stats.fisher_exact(table, alternative="two-sided").pvalue
    p_values.append(p)
    significance.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

df["p-value"] = p_values
df["p-value_formatted"] = df["p-value"].apply(lambda x: f"{x:.3f}")
df["Significance"] = significance

print("=" * 60)
print("STATISTICAL VALIDATION RESULTS (Fisher's Exact Test)")
print("=" * 60)
cols = ["cluster", col_65, col_34, "dominant_group", "p-value_formatted", "Significance"]
print(df[cols].to_string(index=False))

save_path = FIG_DIR / "table4_cluster_summary_with_pvalue.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\np-value 추가 표 저장: {save_path}")

# ── 2. 분류 성능 이항검정 ─────────────────────────────────────────────────────
pred = pd.read_csv(FIG_DIR / "test_predictions.csv")
y_true, y_pred = pred["y_true"].values, pred["y_pred"].values

n = len(y_true)
k = int(np.sum(y_pred == y_true))
acc = k / n
baseline = np.bincount(y_true).max() / n

res = binomtest(k=k, n=n, p=baseline, alternative="greater")

print("\n" + "=" * 60)
print("XGBoost 분류 성능 통계검증")
print("=" * 60)
print(f"테스트 데이터 수: {n}")
print(f"모델 정답 수: {k} (정확도 {acc*100:.2f}%)")
print(f"다수 클래스 베이스라인: {baseline*100:.2f}%")
print(f"정확도 차이: {(acc-baseline)*100:.2f}%p")
print(f"이항검정 p-value: {res.pvalue:.4e}")

if res.pvalue < 0.05:
    print("모델 정확도가 베이스라인보다 통계적으로 유의하게 높습니다.")
else:
    print("모델 정확도가 베이스라인보다 유의하게 높다고 보기 어렵습니다.")

print("\n논문 작성용 문장:")
print(
    f"XGBoost 분류 모델은 {acc*100:.2f}%의 정확도를 보였으며, "
    f"이는 다수 클래스 기준 베이스라인 정확도 {baseline*100:.2f}%보다 "
    f"{(acc-baseline)*100:.2f}%p 높았다. "
    f"단측 이항검정 결과, p-value는 {res.pvalue:.4e}로 나타났다."
)
