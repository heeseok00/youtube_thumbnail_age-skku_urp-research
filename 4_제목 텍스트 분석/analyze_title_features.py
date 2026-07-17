"""
제목 텍스트 분석 파이프라인
- subcategory: 시사/뉴스/사건
- 연령 그룹: 젊은 시청자 채널(~34세 비율 상위33%) vs 고령 시청자 채널(65+ 비율 하위33%)
- 피처: char_count, word_count, bait_punct, jamo_ratio 등 18종
- 분류기: RandomForest, XGBoost, LightGBM + SHAP
- 임베딩: KLUE-RoBERTa 768dim + PCA 64dim + UMAP 64dim
"""

import os, re, json, unicodedata
import numpy as np
import pandas as pd
from collections import Counter

# ── 의존성 체크 ──────────────────────────────────────────────────────────────
def check_and_install():
    pkgs = ["scikit-learn", "xgboost", "lightgbm", "shap", "transformers",
            "torch", "umap-learn", "matplotlib", "seaborn"]
    import subprocess, sys
    for p in pkgs:
        try:
            __import__(p.replace("-", "_").split(".")[0])
        except ImportError:
            print(f"  설치 중: {p}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

print("의존성 확인 중...")
check_and_install()
print("완료.\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import shap

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 폰트
for font in fm.findSystemFonts():
    if any(k in font.lower() for k in ["nanum", "malgun", "notosans", "gothic"]):
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

SUBCATEGORY = "시사/뉴스/사건"
DATA_PATH   = "/home/urp_jwl2/26-1_URP/Data/SOCIETY/SOCIETY_new_category_v3.csv"

# ── 1. 데이터 로드 및 그룹 레이블링 ──────────────────────────────────────────
print("=" * 60)
print("1. 데이터 로드")
df_all = pd.read_csv(DATA_PATH)
df = df_all[df_all["subcategory"] == SUBCATEGORY].copy()
print(f"   {SUBCATEGORY}: {len(df):,}건")

young_cols = ["age_~17", "age_18~24", "age_25~34"]
old_col    = "age_65~"
df["young"] = df[young_cols].sum(axis=1)
df["old"]   = df[old_col]
df["young_score"] = df["young"] / (df["young"] + df["old"])

p33 = df["young_score"].quantile(0.33)
p67 = df["young_score"].quantile(0.67)
young_df = df[df["young_score"] >= p67].copy(); young_df["label"] = 1
old_df   = df[df["young_score"] <= p33].copy(); old_df["label"]   = 0

data = pd.concat([young_df, old_df], ignore_index=True)
print(f"   젊은 그룹(label=1): {len(young_df):,}건 | 고령 그룹(label=0): {len(old_df):,}건")
print(f"   젊은 그룹 ~34세 평균: {young_df['young'].mean():.1f}%  65+ 평균: {young_df['old'].mean():.1f}%")
print(f"   고령 그룹 ~34세 평균: {old_df['young'].mean():.1f}%  65+ 평균: {old_df['old'].mean():.1f}%")


# ── 2. 제목 피처 추출 ─────────────────────────────────────────────────────────
print("\n2. 제목 피처 추출")

BAIT_PUNCT   = set("!?#")
SUPERLATIVES = ["가장","최고","최대","최저","역대","극도","완전","진짜","정말","절대","엄청","초","mega","역사상"]
PERSONAL_PRO = ["나","나는","나의","내가","저","저는","우리","그들","당신","여러분","우리가"]
DEMO_PRO     = ["이것","그것","저것","이게","그게","저게","이거","그거","저거","이런","그런","저런"]

def extract_features(title: str) -> dict:
    if not isinstance(title, str) or title.strip() == "":
        return {k: 0 for k in [
            "char_count","word_count","mean_word_length","ttr",
            "punctuation_ratio","bait_punct_count","non_bait_punct_count",
            "ellipsis_count","emoji_count","numbers_count",
            "has_list_number","has_money_amount",
            "personal_pronoun_count","demonstrative_count","superlative_count",
            "jamo_ratio","exclaim_count","question_count"
        ]}

    title = title.strip()
    words = title.split()
    chars = [c for c in title if not c.isspace()]

    # 자모 비율 (ㅋㅋ, ㅠㅠ 등)
    jamo = [c for c in title if unicodedata.name(c, "").startswith("HANGUL LETTER")]
    # 이모지
    emoji = [c for c in title if unicodedata.category(c) in ("So", "Sm") or
             ('\U0001F300' <= c <= '\U0001FAFF')]
    # 구두점
    puncts = [c for c in title if unicodedata.category(c).startswith("P") or c in "!?#"]
    bait_p = [c for c in puncts if c in BAIT_PUNCT]

    # 숫자 덩어리
    num_tokens = re.findall(r"\d[\d,\.]*", title)
    has_money  = 1 if re.search(r"만원|억|천만|달러|\$|원|￦", title) else 0
    has_list_n = 1 if re.search(r"\b\d+가지|\b\d+위|\b\d+개|\b\d+등", title) else 0

    personal = sum(1 for p in PERSONAL_PRO if p in title)
    demo     = sum(1 for p in DEMO_PRO      if p in title)
    superl   = sum(1 for p in SUPERLATIVES  if p in title)

    ttr = len(set(words)) / len(words) if words else 0
    mean_wl = np.mean([len(w) for w in words]) if words else 0
    punct_ratio = len(puncts) / len(chars) if chars else 0

    return {
        "char_count":             len(chars),
        "word_count":             len(words),
        "mean_word_length":       round(mean_wl, 3),
        "ttr":                    round(ttr, 3),
        "punctuation_ratio":      round(punct_ratio, 4),
        "bait_punct_count":       len(bait_p),
        "non_bait_punct_count":   len(puncts) - len(bait_p),
        "ellipsis_count":         title.count("…") + title.count("..."),
        "emoji_count":            len(emoji),
        "numbers_count":          len(num_tokens),
        "has_list_number":        has_list_n,
        "has_money_amount":       has_money,
        "personal_pronoun_count": personal,
        "demonstrative_count":    demo,
        "superlative_count":      superl,
        "jamo_ratio":             round(len(jamo) / len(chars), 4) if chars else 0,
        "exclaim_count":          title.count("!"),
        "question_count":         title.count("?"),
    }

feat_list = [extract_features(t) for t in data["title"]]
feat_df   = pd.DataFrame(feat_list, index=data.index)
FEATURE_COLS = feat_df.columns.tolist()

print(f"   피처 수: {len(FEATURE_COLS)}")
print(f"   피처 목록: {FEATURE_COLS}")

# 원본 + 그룹 레이블 + 18개 피처를 하나의 CSV로 저장
data_out = data.copy()
data_out["age_group"] = data_out["label"].map({1: "젊은_채널", 0: "고령_채널"})
for col in FEATURE_COLS:
    data_out[col] = feat_df[col]
data_out = data_out.drop(columns=["young", "old", "young_score", "label"])
OUT_CSV = os.path.join(OUTPUT_DIR, "시사뉴스사건_분석데이터.csv")
data_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"   저장 완료: 시사뉴스사건_분석데이터.csv ({len(data_out):,}행, {len(data_out.columns)}컬럼)")


# ── 3. ML 분류기 학습 + SHAP ──────────────────────────────────────────────────
print("\n3. ML 분류기 학습")

X = feat_df[FEATURE_COLS].values
y = data["label"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "roc_auc"]

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost":      xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                       eval_metric="logloss", random_state=42, verbosity=0),
    "LightGBM":     lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                        random_state=42, verbose=-1),
}

results = {}
for name, model in models.items():
    print(f"   [{name}] 학습 중...")
    cv_res = cross_validate(model, X_sc, y, cv=cv, scoring=scoring, return_estimator=True)
    results[name] = {
        "accuracy": cv_res["test_accuracy"],
        "f1":       cv_res["test_f1"],
        "roc_auc":  cv_res["test_roc_auc"],
        "estimators": cv_res["estimator"],
    }
    print(f"     Accuracy : {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
    print(f"     F1       : {cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}")
    print(f"     ROC-AUC  : {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")

# 결과 저장
metrics_rows = []
for name, res in results.items():
    metrics_rows.append({
        "모델": name,
        "Accuracy":  f"{res['accuracy'].mean():.4f} ± {res['accuracy'].std():.4f}",
        "F1":        f"{res['f1'].mean():.4f} ± {res['f1'].std():.4f}",
        "ROC-AUC":   f"{res['roc_auc'].mean():.4f} ± {res['roc_auc'].std():.4f}",
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "ml_metrics.csv"), index=False, encoding="utf-8-sig")
print(f"\n   metrics 저장: ml_metrics.csv")


# ── 4. SHAP ───────────────────────────────────────────────────────────────────
print("\n4. SHAP 분석")

def shap_for_group(model, X_group, feature_names, title, out_path):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_group)
    if isinstance(sv, list):
        sv = sv[1]
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(sv, X_group, feature_names=feature_names, show=False)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return sv

# XGBoost 사용 (대표 모델)
best_xgb = results["XGBoost"]["estimators"][0]
best_xgb.fit(X_sc, y)

X_young = X_sc[y == 1]
X_old   = X_sc[y == 0]

sv_young = shap_for_group(best_xgb, X_young, FEATURE_COLS,
                           "SHAP — 젊은 시청자 채널 (~34세)", 
                           os.path.join(OUTPUT_DIR, "shap_young.png"))
sv_old   = shap_for_group(best_xgb, X_old,   FEATURE_COLS,
                           "SHAP — 고령 시청자 채널 (65+)",
                           os.path.join(OUTPUT_DIR, "shap_old.png"))
print("   shap_young.png, shap_old.png 저장 완료")

# 피처 중요도 저장
importance = pd.DataFrame({
    "feature":        FEATURE_COLS,
    "shap_young_mean": np.abs(sv_young).mean(axis=0),
    "shap_old_mean":   np.abs(sv_old).mean(axis=0),
})
importance.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False, encoding="utf-8-sig")


# ── 5. KLUE-RoBERTa 임베딩 ────────────────────────────────────────────────────
print("\n5. KLUE-RoBERTa 임베딩 분류")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import umap

    MODEL_NAME = "klue/roberta-base"
    print(f"   모델 로드: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rob_model  = AutoModel.from_pretrained(MODEL_NAME)
    rob_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rob_model.to(device)
    print(f"   device: {device}")

    BATCH = 64
    def encode_titles(titles):
        all_emb = []
        for i in range(0, len(titles), BATCH):
            batch = list(titles[i:i+BATCH])
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = rob_model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(cls)
            if (i // BATCH) % 10 == 0:
                print(f"     임베딩 {i+len(batch)}/{len(titles)}")
        return np.vstack(all_emb)

    titles = data["title"].fillna("").tolist()
    print(f"   임베딩 생성 중 ({len(titles):,}건)...")
    X_emb = encode_titles(titles)
    np.save(os.path.join(OUTPUT_DIR, "embeddings_768.npy"), X_emb)
    print(f"   embeddings_768.npy 저장")

    emb_results = {}
    for dim_name, X_dim in [
        ("원본 768차원",  X_emb),
        ("PCA 64차원",    PCA(n_components=64, random_state=42).fit_transform(X_emb)),
    ]:
        try:
            red = umap.UMAP(n_components=64, random_state=42, n_jobs=1).fit_transform(X_emb)
            dims = [("원본 768차원", X_emb), ("PCA 64차원", PCA(n_components=64, random_state=42).fit_transform(X_emb)), ("UMAP 64차원", red)]
        except Exception:
            dims = [("원본 768차원", X_emb), ("PCA 64차원", PCA(n_components=64, random_state=42).fit_transform(X_emb))]
        break

    emb_rows = []
    for dim_name, X_dim in dims:
        X_dim_sc = StandardScaler().fit_transform(X_dim)
        for mname, model in models.items():
            cv_res = cross_validate(model, X_dim_sc, y, cv=cv, scoring=scoring)
            emb_rows.append({
                "차원": dim_name, "모델": mname,
                "Accuracy": f"{cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}",
                "F1":       f"{cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}",
                "ROC-AUC":  f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
            })
            print(f"   [{dim_name} / {mname}] ROC-AUC: {cv_res['test_roc_auc'].mean():.4f}")

    emb_df = pd.DataFrame(emb_rows)
    emb_df.to_csv(os.path.join(OUTPUT_DIR, "embedding_metrics.csv"), index=False, encoding="utf-8-sig")
    print("   embedding_metrics.csv 저장 완료")

    # UMAP 2D scatter 시각화
    try:
        X_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(X_emb)
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#ef4444" if yi == 1 else "#3b82f6" for yi in y]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=5, alpha=0.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#ef4444", label="젊은 채널 (~34세)"),
                            Patch(color="#3b82f6", label="고령 채널 (65+)")])
        ax.set_title("UMAP 2D — KLUE-RoBERTa 임베딩", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "umap_scatter.png"), dpi=150)
        plt.close()
        print("   umap_scatter.png 저장 완료")
    except Exception as e:
        print(f"   UMAP 2D 실패: {e}")

except Exception as e:
    print(f"   임베딩 단계 오류: {e}")
    emb_df = pd.DataFrame()

# ── 6. 최종 요약 출력 ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("분석 완료. 생성된 파일:")
for f in ["ml_metrics.csv", "shap_importance.csv", "shap_young.png",
          "shap_old.png", "embedding_metrics.csv", "umap_scatter.png"]:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        print(f"  ✓ {f}")
print("=" * 60)
