"""Step 2 — 캡션 의미 분석: TF-IDF → XGBoost → SHAP → Sentence-BERT → K-Means.

예나의 코랩 버전을 서버(urp_yena 환경)용으로 튜닝.
- 입력: outputs/diet_sample_2500_vlm.csv (Step 1 출력)
- 변경점: 코랩 경로/설치 제거, ERROR 캡션 필터 추가,
  Step 3 통계검증이 독립 실행되도록 테스트 예측 결과를 CSV로 저장
- 출력: outputs/figures/ 에 이미지 4개 + 표 3개 + test_predictions.csv

실행: python 02_caption_analysis.py
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib  # noqa: F401  (한글 폰트)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

STAGE_DIR = Path(__file__).resolve().parent
INPUT_CSV = STAGE_DIR / "outputs/diet_sample_2500_vlm.csv"
OUTPUT_DIR = STAGE_DIR / "outputs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def out(name):
    return str(OUTPUT_DIR / name)


C_34 = "#4A90D9"
C_65 = "#E8734A"
PALETTE_CLUSTER = [
    "#E07B54", "#5B9BD5", "#70AD47", "#FFC000",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#3498DB",
    "#2ECC71", "#F39C12", "#8E44AD", "#16A085",
]
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1. Loading Data")
print("=" * 60)

df = pd.read_csv(INPUT_CSV, low_memory=False)
df = df[["vlm_caption", "target"]].dropna()
df = df[~df["vlm_caption"].astype(str).str.startswith("ERROR")]  # 실패 캡션 제외
df["target"] = df["target"].astype(int)
print(f"  Total: {len(df)} | ~34세(0): {(df.target==0).sum()} | 65+(1): {(df.target==1).sum()}")

# ── 2. TF-IDF ────────────────────────────────────────────────────────────────
print("\nSTEP 2. TF-IDF Vectorization")
tfidf = TfidfVectorizer(stop_words="english", max_features=2000, min_df=3, ngram_range=(1, 1))
X = tfidf.fit_transform(df["vlm_caption"])
y = df["target"].values
feature_names = np.array(tfidf.get_feature_names_out())
print(f"  Vocabulary size: {len(feature_names)}")

# ── 3. XGBoost ───────────────────────────────────────────────────────────────
print("\nSTEP 3. XGBoost Training")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, verbosity=0,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["~34세", "65+"]))

# Step 3(통계검증) 독립 실행용으로 테스트 예측 저장
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    out("test_predictions.csv"), index=False
)

# ── 4. SHAP ──────────────────────────────────────────────────────────────────
print("\nSTEP 4. SHAP Analysis")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_train.toarray())
mean_shap = np.mean(shap_vals, axis=0)

TOP_N = 100
top_65_idx = np.argsort(mean_shap)[-TOP_N:]
top_34_idx = np.argsort(mean_shap)[:TOP_N]
top_65_words, top_65_vals = feature_names[top_65_idx], mean_shap[top_65_idx]
top_34_words, top_34_vals = feature_names[top_34_idx], mean_shap[top_34_idx]
print(f"  65+ top words: {list(top_65_words[-10:])}")
print(f"  ~34 top words: {list(top_34_words[:10])}")

all_words = list(set(list(top_65_words) + list(top_34_words)))
all_shap_dict = {w: mean_shap[np.where(feature_names == w)[0][0]] for w in all_words}

# ── 5. Sentence-BERT 임베딩 ───────────────────────────────────────────────────
print("\nSTEP 5. Sentence-BERT Embedding")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
valid_words = sorted(all_words)
emb_matrix = embedder.encode(
    valid_words, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
)
shap_arr = np.array([all_shap_dict[w] for w in valid_words])
group_arr = np.array(["65+" if s > 0 else "~34" for s in shap_arr])
print(f"  Words with embedding: {len(valid_words)}")

# ── 6. Elbow → K 결정 ────────────────────────────────────────────────────────
print("\nSTEP 6. Finding Optimal K")
inertias, K_range = [], range(3, 12)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(emb_matrix)
    inertias.append(km.inertia_)
diffs2 = np.diff(np.diff(inertias))
optimal_k = list(K_range)[np.argmin(diffs2) + 1]
optimal_k = max(5, min(optimal_k, 8))
print(f"  Optimal K: {optimal_k}")

# ── 7. K-Means ───────────────────────────────────────────────────────────────
print("\nSTEP 7. K-Means Clustering")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(emb_matrix)

wc_df = pd.DataFrame({
    "word": valid_words,
    "cluster": cluster_labels,
    "shap": shap_arr,
    "group": group_arr,
    "abs_shap": np.abs(shap_arr),
})

cluster_rep = {}
for c in range(optimal_k):
    sub = wc_df[wc_df.cluster == c].sort_values("abs_shap", ascending=False)
    n65, n34 = (sub.group == "65+").sum(), (sub.group == "~34").sum()
    cluster_rep[c] = {"words": sub.head(8)["word"].tolist(), "n65": n65, "n34": n34}
    print(f"  Cluster {c} | 65+:{n65}  ~34:{n34} | {cluster_rep[c]['words']}")

# ── FIGURE 1. SHAP Butterfly ─────────────────────────────────────────────────
print("\nGenerating Figure 1: SHAP Butterfly Chart")
SHOW = 20
bf_65_words, bf_65_vals = top_65_words[-SHOW:], top_65_vals[-SHOW:]
bf_34_words, bf_34_vals = top_34_words[:SHOW], top_34_vals[:SHOW]

fig1, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)
fig1.patch.set_facecolor("#F8F9FA")

ax_l = axes[0]
si = np.argsort(bf_34_vals)
words_l, vals_l = bf_34_words[si], bf_34_vals[si]
bars_l = ax_l.barh(range(SHOW), np.abs(vals_l), color=C_34, alpha=0.85, height=0.7)
ax_l.set_yticks(range(SHOW)); ax_l.set_yticklabels(words_l, fontsize=10)
ax_l.invert_xaxis(); ax_l.set_xlabel("|Mean SHAP Value|", fontsize=11)
ax_l.set_title("~34세 연령대\n특징 키워드", fontsize=13, color=C_34, fontweight="bold", pad=12)
ax_l.set_facecolor("#F8F9FA")
for bar, val in zip(bars_l, np.abs(vals_l)):
    ax_l.text(val + 0.0003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
              va="center", ha="right", fontsize=7.5, color="#555")

ax_r = axes[1]
si2 = np.argsort(bf_65_vals)
words_r, vals_r = bf_65_words[si2], bf_65_vals[si2]
bars_r = ax_r.barh(range(SHOW), vals_r, color=C_65, alpha=0.85, height=0.7)
ax_r.set_yticks(range(SHOW)); ax_r.set_yticklabels(words_r, fontsize=10)
ax_r.set_xlabel("Mean SHAP Value", fontsize=11)
ax_r.set_title("65세 이상 연령대\n특징 키워드", fontsize=13, color=C_65, fontweight="bold", pad=12)
ax_r.set_facecolor("#F8F9FA")
for bar, val in zip(bars_r, vals_r):
    ax_r.text(val + 0.0003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
              va="center", ha="left", fontsize=7.5, color="#555")

fig1.suptitle("SHAP Butterfly Chart: 연령대별 썸네일 특징 단어 비교",
              fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(w_pad=3)
fig1.savefig(out("fig1_shap_butterfly.png"), bbox_inches="tight", facecolor="#F8F9FA", dpi=150)
plt.close(fig1)
print(f"  Saved: {out('fig1_shap_butterfly.png')}")

# ── FIGURE 2. 클러스터별 연령대 분포 ─────────────────────────────────────────
print("\nGenerating Figure 2: Cluster Distribution Chart")
fig2, axes2 = plt.subplots(1, optimal_k, figsize=(optimal_k * 3.2, 6))
fig2.patch.set_facecolor("#F8F9FA")
if optimal_k == 1:
    axes2 = [axes2]
for c in range(optimal_k):
    ax = axes2[c]
    info = cluster_rep[c]
    total = info["n65"] + info["n34"]
    pct65, pct34 = info["n65"] / total * 100, info["n34"] / total * 100
    ax.bar([0], [pct65], color=C_65, alpha=0.9, width=0.5)
    ax.bar([0], [pct34], bottom=[pct65], color=C_34, alpha=0.9, width=0.5)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 100); ax.set_xticks([])
    ax.set_ylabel("비율 (%)" if c == 0 else "", fontsize=10)
    ax.set_facecolor("#F8F9FA")
    ax.text(0, pct65 / 2, f"{pct65:.0f}%", ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")
    ax.text(0, pct65 + pct34 / 2, f"{pct34:.0f}%", ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")
    ax.set_title(f"Cluster {c}\n─────────\n" + "\n".join(info["words"][:5]),
                 fontsize=9, pad=8, color=PALETTE_CLUSTER[c], fontweight="bold")

handles = [mpatches.Patch(color=C_65, label="65세 이상"),
           mpatches.Patch(color=C_34, label="~34세 이하")]
fig2.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
            frameon=False, bbox_to_anchor=(0.5, -0.05))
fig2.suptitle("의미 클러스터별 연령대 분포\n(SHAP 선정 단어 기반)",
              fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
fig2.savefig(out("fig2_cluster_distribution.png"), bbox_inches="tight", facecolor="#F8F9FA", dpi=150)
plt.close(fig2)
print(f"  Saved: {out('fig2_cluster_distribution.png')}")

# ── FIGURE 3. 2D Word Map ────────────────────────────────────────────────────
print("\nGenerating Figure 3: 2D Word Map")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(emb_matrix)
var_exp = pca.explained_variance_ratio_ * 100

fig3, ax3 = plt.subplots(figsize=(13, 9))
fig3.patch.set_facecolor("#F8F9FA"); ax3.set_facecolor("#F8F9FA")
for c in range(optimal_k):
    idx_c = np.where(cluster_labels == c)[0]
    ax3.scatter(coords[idx_c, 0], coords[idx_c, 1], color=PALETTE_CLUSTER[c],
                s=120, alpha=0.7, edgecolors="white", linewidths=0.8, zorder=3)
for i, (word, grp, sv) in enumerate(zip(valid_words, group_arr, shap_arr)):
    ax3.scatter(coords[i, 0], coords[i, 1], marker="^" if grp == "65+" else "v",
                s=60, color=C_65 if grp == "65+" else C_34, alpha=0.9,
                edgecolors="white", linewidths=0.6, zorder=4)
for _, row in wc_df.nlargest(50, "abs_shap").iterrows():
    idx = valid_words.index(row["word"])
    ax3.annotate(row["word"], (coords[idx, 0], coords[idx, 1]), fontsize=8.5,
                 xytext=(4, 4), textcoords="offset points",
                 color="#333333", fontweight="bold")

cluster_handles = [mpatches.Patch(color=PALETTE_CLUSTER[c], label=f"Cluster {c}")
                   for c in range(optimal_k)]
group_handles = [
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_65,
               markersize=10, label="65세 이상 기여 단어"),
    plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=C_34,
               markersize=10, label="~34세 기여 단어"),
]
ax3.legend(handles=cluster_handles + group_handles, loc="upper right",
           fontsize=9, framealpha=0.9, ncol=2)
ax3.set_xlabel(f"PCA 1 ({var_exp[0]:.1f}% explained)", fontsize=11)
ax3.set_ylabel(f"PCA 2 ({var_exp[1]:.1f}% explained)", fontsize=11)
ax3.set_title("SHAP 선정 키워드의 의미 공간\n(Sentence-BERT 임베딩 + PCA 2D)",
              fontsize=14, fontweight="bold", pad=14)
ax3.axhline(0, color="#cccccc", linewidth=0.5, linestyle="--", zorder=1)
ax3.axvline(0, color="#cccccc", linewidth=0.5, linestyle="--", zorder=1)
plt.tight_layout()
fig3.savefig(out("fig3_word_map.png"), bbox_inches="tight", facecolor="#F8F9FA", dpi=150)
plt.close(fig3)
print(f"  Saved: {out('fig3_word_map.png')}")

# ── FIGURE 4. Elbow ──────────────────────────────────────────────────────────
print("\nGenerating Figure 4: Elbow Curve")
fig4, ax4 = plt.subplots(figsize=(7, 4))
fig4.patch.set_facecolor("#F8F9FA"); ax4.set_facecolor("#F8F9FA")
ax4.plot(list(K_range), inertias, "o-", color="#4A90D9", linewidth=2, markersize=8)
ax4.axvline(optimal_k, color=C_65, linewidth=1.5, linestyle="--",
            label=f"Optimal K = {optimal_k}")
ax4.set_xlabel("K (클러스터 수)", fontsize=11)
ax4.set_ylabel("Inertia", fontsize=11)
ax4.set_title("K-Means Elbow Curve", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
plt.tight_layout()
fig4.savefig(out("fig4_elbow.png"), bbox_inches="tight", facecolor="#F8F9FA", dpi=150)
plt.close(fig4)
print(f"  Saved: {out('fig4_elbow.png')}")

# ── 요약 출력 및 표 저장 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLUSTER SUMMARY TABLE")
print("=" * 60)
for c in range(optimal_k):
    sub = wc_df[wc_df.cluster == c].sort_values("abs_shap", ascending=False)
    info = cluster_rep[c]
    total = info["n65"] + info["n34"]
    dom = "65+" if info["n65"] > info["n34"] else "~34"
    pct = max(info["n65"], info["n34"]) / total * 100
    print(f"\n  [Cluster {c}] 지배 그룹: {dom} ({pct:.0f}%)")
    print(f"  65+ 단어: {sub[sub.group == '65+'].head(5)['word'].tolist()}")
    print(f"  ~34 단어: {sub[sub.group == '~34'].head(5)['word'].tolist()}")

keyword_df = pd.DataFrame({
    "word": list(top_65_words[::-1]) + list(top_34_words),
    "direction": ["65+"] * len(top_65_words) + ["~34"] * len(top_34_words),
    "mean_shap": list(top_65_vals[::-1]) + list(top_34_vals),
})
keyword_df["abs_mean_shap"] = keyword_df["mean_shap"].abs()
keyword_df["rank_within_direction"] = (
    keyword_df.groupby("direction")["abs_mean_shap"]
    .rank(method="first", ascending=False).astype(int)
)
keyword_df = keyword_df.sort_values(["direction", "rank_within_direction"])
keyword_df.to_csv(out("table1_age_keyword_shap.csv"), index=False, encoding="utf-8-sig")

wc_df.sort_values(["cluster", "abs_shap"], ascending=[True, False]).to_csv(
    out("table2_cluster_keywords.csv"), index=False, encoding="utf-8-sig"
)

rows = []
for c in range(optimal_k):
    sub = wc_df[wc_df.cluster == c]
    n65, n34 = int((sub.group == "65+").sum()), int((sub.group == "~34").sum())
    total = n65 + n34
    rows.append({
        "cluster": c,
        "n_65plus_keywords": n65,
        "n_under34_keywords": n34,
        "total_keywords": total,
        "pct_65plus": n65 / total * 100 if total else 0,
        "pct_under34": n34 / total * 100 if total else 0,
        "dominant_group": "65+" if n65 > n34 else "~34",
        "dominant_pct": max(n65, n34) / total * 100 if total else 0,
        "top_keywords": ", ".join(
            sub.sort_values("abs_shap", ascending=False).head(10)["word"]
        ),
    })
pd.DataFrame(rows).to_csv(out("table3_cluster_summary.csv"), index=False, encoding="utf-8-sig")

print(f"\n완료. 모든 산출물: {OUTPUT_DIR}")
