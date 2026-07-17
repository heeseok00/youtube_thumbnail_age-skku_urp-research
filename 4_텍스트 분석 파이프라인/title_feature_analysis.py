"""
제목(title) 피처 기반 연령대 분류 분석
시사/뉴스/사건 subcategory — 젊은 시청자 vs 고령 시청자 채널

Usage:
  python title_feature_analysis.py              # 전체 실행 (피처추출 + ML + SHAP)
  python title_feature_analysis.py --no-embed   # KLUE-RoBERTa 임베딩 제외
  python title_feature_analysis.py --embed-only # 임베딩 분류만 실행
"""

import re, os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/home/urp_jwl2/.local/share/fonts/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

DATA_PATH   = "/home/urp_jwl2/26-1_URP/Data/SOCIETY/SOCIETY_new_category_v3.csv"
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
SUBCATEGORY = "시사/뉴스/사건"

# ── 1. 데이터 로드 & 그룹 레이블 ────────────────────────────────────────────────
def load_data():
    print("[1/4] 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)
    sub = df[df["subcategory"] == SUBCATEGORY].copy()

    age_young = ["age_~17", "age_18~24", "age_25~34"]
    sub["_young"] = sub[age_young].sum(axis=1)
    sub["_old"]   = sub["age_65~"].fillna(0)
    sub["_score"] = sub["_young"] / (sub["_young"] + sub["_old"] + 1e-9)

    p33 = sub["_score"].quantile(0.33)
    p67 = sub["_score"].quantile(0.67)

    young_df = sub[sub["_score"] >= p67].copy()
    old_df   = sub[sub["_score"] <= p33].copy()

    young_df["label"] = 0   # 0 = 젊은 시청자
    old_df["label"]   = 1   # 1 = 고령 시청자

    merged = pd.concat([young_df, old_df]).dropna(subset=["title"]).reset_index(drop=True)
    print(f"    젊은 그룹: {(merged['label']==0).sum()}건  |  고령 그룹: {(merged['label']==1).sum()}건")
    return merged

# ── 2. 피처 추출 ─────────────────────────────────────────────────────────────────
BAIT_PUNCT    = set("!\"?#")
NON_BAIT_PUNCT = set(".,;:-/")
PERSONAL_PRONOUN = re.compile(r"나|너|우리|당신|저|제|내가|네가|우린|저희")
DEMONSTRATIVE    = re.compile(r"이것|그것|저것|이게|그게|저게|이거|그거|저거|여기|거기|저기")
SUPERLATIVE      = re.compile(r"가장|최고|최대|최초|역대|세계|국내|압도적")
JAMO_RE          = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
EMOJI_RE         = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF]+", flags=re.UNICODE
)
NUMBER_RE   = re.compile(r"\d+")
LIST_NUM_RE = re.compile(r"\d+(가지|위|번|개|명|종|단계|포인트|tip|tips)", re.I)
MONEY_RE    = re.compile(r"\d+(만원|억|천원|원|달러|\$|￦|USD|KRW)", re.I)
ELLIPSIS_RE = re.compile(r"\.{2,}|…")


def extract_features(title: str) -> dict:
    t = str(title)
    no_space = t.replace(" ", "")
    words = t.split()

    char_count       = len(no_space)
    word_count       = len(words)
    mean_word_length = np.mean([len(w) for w in words]) if words else 0
    unique_chars     = len(set(no_space))
    ttr              = unique_chars / char_count if char_count > 0 else 0

    punct_count        = sum(1 for c in t if not c.isalnum() and not c.isspace())
    punctuation_ratio  = punct_count / len(t) if len(t) > 0 else 0
    bait_punct_count   = sum(1 for c in t if c in BAIT_PUNCT)
    non_bait_punct_count = sum(1 for c in t if c in NON_BAIT_PUNCT)
    ellipsis_count     = len(ELLIPSIS_RE.findall(t))
    emoji_count        = len(EMOJI_RE.findall(t))

    numbers_count      = len(NUMBER_RE.findall(t))
    has_list_number    = int(bool(LIST_NUM_RE.search(t)))
    has_money_amount   = int(bool(MONEY_RE.search(t)))

    personal_pronoun_count = len(PERSONAL_PRONOUN.findall(t))
    demonstrative_count    = len(DEMONSTRATIVE.findall(t))
    superlative_count      = len(SUPERLATIVE.findall(t))
    jamo_ratio = len(JAMO_RE.findall(t)) / len(t) if len(t) > 0 else 0

    return {
        "char_count":            char_count,
        "word_count":            word_count,
        "mean_word_length":      mean_word_length,
        "ttr":                   ttr,
        "punctuation_ratio":     punctuation_ratio,
        "bait_punct_count":      bait_punct_count,
        "non_bait_punct_count":  non_bait_punct_count,
        "ellipsis_count":        ellipsis_count,
        "emoji_count":           emoji_count,
        "numbers_count":         numbers_count,
        "has_list_number":       has_list_number,
        "has_money_amount":      has_money_amount,
        "personal_pronoun_count": personal_pronoun_count,
        "demonstrative_count":   demonstrative_count,
        "superlative_count":     superlative_count,
        "jamo_ratio":            jamo_ratio,
    }

FEATURE_COLS = list(extract_features("test").keys())

FEATURE_LABELS = {
    "char_count":             "char_count (문자수)",
    "word_count":             "word_count (어절수)",
    "mean_word_length":       "mean_word_length (평균어절길이)",
    "ttr":                    "ttr (어휘다양성)",
    "punctuation_ratio":      "punctuation_ratio (구두점비율)",
    "bait_punct_count":       "bait_punct_count (!\"?#)",
    "non_bait_punct_count":   "non_bait_punct_count (.,;:-)",
    "ellipsis_count":         "ellipsis_count (말줄임표)",
    "emoji_count":            "emoji_count (이모지)",
    "numbers_count":          "numbers_count (숫자)",
    "has_list_number":        "has_list_number (리스트숫자)",
    "has_money_amount":       "has_money_amount (금액표현)",
    "personal_pronoun_count": "personal_pronoun_count (인칭대명사)",
    "demonstrative_count":    "demonstrative_count (지시대명사)",
    "superlative_count":      "superlative_count (최상급)",
    "jamo_ratio":             "jamo_ratio (자모비율)",
}

# ── 3. ML 분류 ───────────────────────────────────────────────────────────────────
def run_ml(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import shap

    print("\n[2/4] ML 분류 실행 중...")

    feat_df = pd.DataFrame(list(df["title"].apply(extract_features)))
    X = feat_df[FEATURE_COLS].values
    y = df["label"].values

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":      XGBClassifier(n_estimators=200, random_state=42,
                                       eval_metric="logloss", verbosity=0),
        "LightGBM":     LGBMClassifier(n_estimators=200, random_state=42,
                                        verbose=-1, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc"]
    results = {}

    for name, clf in models.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "Accuracy": (scores["test_accuracy"].mean(),  scores["test_accuracy"].std()),
            "F1":       (scores["test_f1"].mean(),        scores["test_f1"].std()),
            "ROC-AUC":  (scores["test_roc_auc"].mean(),   scores["test_roc_auc"].std()),
        }
        print(f"    [{name}]  Acc {results[name]['Accuracy'][0]:.4f}  "
              f"F1 {results[name]['F1'][0]:.4f}  AUC {results[name]['ROC-AUC'][0]:.4f}")

    # ── SHAP (최고 성능 모델로) ───────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"][0])
    print(f"\n    SHAP 계산: {best_name}...")
    best_clf = models[best_name]
    best_clf.fit(X, y)

    explainer  = shap.TreeExplainer(best_clf)
    shap_vals  = explainer.shap_values(X)

    if isinstance(shap_vals, list):          # RF returns list [class0, class1]
        shap_for_plot = shap_vals[1]
    else:
        shap_for_plot = shap_vals

    feat_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax_idx, (label_val, group_name) in enumerate([(0, "젊은 시청자 (~34세)"), (1, "고령 시청자 (65+)")]):
        mask     = y == label_val
        sv_group = shap_for_plot[mask]
        xg       = X[mask]

        mean_abs = np.abs(sv_group).mean(axis=0)
        order    = np.argsort(mean_abs)[::-1][:16][::-1]

        ax = axes[ax_idx]
        colors = ["#d73027" if sv_group[:, i].mean() > 0 else "#4575b4"
                  for i in order]
        ax.barh(range(len(order)), mean_abs[order], color=colors, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_labels[i] for i in order], fontsize=9)
        ax.set_xlabel("mean |SHAP value|", fontsize=10)
        ax.set_title(f"{group_name}\n({best_name})", fontsize=12)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"시사/뉴스/사건 — 연령 그룹별 SHAP 피처 중요도\n(모델: {best_name})", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    SHAP 그래프 저장: {out_path}")

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":    name,
            "Accuracy": f"{res['Accuracy'][0]:.4f} ± {res['Accuracy'][1]:.4f}",
            "F1":       f"{res['F1'][0]:.4f} ± {res['F1'][1]:.4f}",
            "ROC-AUC":  f"{res['ROC-AUC'][0]:.4f} ± {res['ROC-AUC'][1]:.4f}",
        })
    result_df = pd.DataFrame(rows)
    result_path = os.path.join(OUTPUT_DIR, "ml_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"    ML 결과 저장: {result_path}")
    print()
    print(result_df.to_string(index=False))

    return results, feat_df, y, best_clf, shap_for_plot, X

# ── 4. KLUE-RoBERTa 임베딩 분류 ──────────────────────────────────────────────────
def run_embedding(df):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    print("\n[3/4] KLUE-RoBERTa 임베딩 추출 중...")

    MODEL_NAME = "klue/roberta-base"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"    디바이스: {device}")

    titles = df["title"].tolist()
    y      = df["label"].values
    embeddings = []

    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(titles), BATCH):
            batch = titles[i:i+BATCH]
            enc   = tokenizer(batch, padding=True, truncation=True,
                              max_length=64, return_tensors="pt").to(device)
            out   = model(**enc)
            cls   = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if i % (BATCH * 10) == 0:
                print(f"    {i}/{len(titles)} 완료...")

    emb_raw = np.vstack(embeddings)
    print(f"    임베딩 완료: {emb_raw.shape}")

    emb_path = os.path.join(OUTPUT_DIR, "klue_embeddings.npy")
    np.save(emb_path, emb_raw)
    np.save(os.path.join(OUTPUT_DIR, "klue_labels.npy"), y)
    print(f"    임베딩 저장: {emb_path}")

    scaler  = StandardScaler()
    emb_std = scaler.fit_transform(emb_raw)

    pca   = PCA(n_components=64, random_state=42)
    emb_pca = pca.fit_transform(emb_std)

    try:
        reducer  = umap.UMAP(n_components=64, random_state=42, n_jobs=1)
        emb_umap = reducer.fit_transform(emb_std)
        umap_ok  = True
    except ImportError:
        print("    umap-learn 미설치, UMAP 건너뜀")
        umap_ok = False

    configs = [("원본 768차원", emb_std)]
    configs.append(("PCA 64차원",  emb_pca))
    if umap_ok:
        configs.append(("UMAP 64차원", emb_umap))

    models_emb = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":      XGBClassifier(n_estimators=200, random_state=42,
                                       eval_metric="logloss", verbosity=0),
        "LightGBM":     LGBMClassifier(n_estimators=200, random_state=42,
                                        verbose=-1, n_jobs=-1),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc"]
    emb_results = {}
    print("\n    [임베딩 분류 결과]")

    for dim_name, X_emb in configs:
        emb_results[dim_name] = {}
        for mname, clf in models_emb.items():
            scores = cross_validate(clf, X_emb, y, cv=cv, scoring=scoring, n_jobs=-1)
            emb_results[dim_name][mname] = {
                "Accuracy": (scores["test_accuracy"].mean(), scores["test_accuracy"].std()),
                "F1":       (scores["test_f1"].mean(),       scores["test_f1"].std()),
                "ROC-AUC":  (scores["test_roc_auc"].mean(),  scores["test_roc_auc"].std()),
            }
            r = emb_results[dim_name][mname]
            print(f"    {dim_name} / {mname}: "
                  f"Acc {r['Accuracy'][0]:.4f}  F1 {r['F1'][0]:.4f}  AUC {r['ROC-AUC'][0]:.4f}")

    # ── 3D UMAP 시각화 ────────────────────────────────────────────────────────
    try:
        reducer3d = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        emb3d     = reducer3d.fit_transform(emb_std)
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")
        colors = ["#d73027" if lbl == 0 else "#4575b4" for lbl in y]
        ax.scatter(emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
                   c=colors, alpha=0.4, s=8)
        ax.set_title("KLUE-RoBERTa UMAP 3D\n빨강=젊은 시청자 / 파랑=고령 시청자", fontsize=11)
        umap_path = os.path.join(OUTPUT_DIR, "umap_3d.png")
        fig.savefig(umap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n    UMAP 3D 저장: {umap_path}")
    except Exception as e:
        print(f"    UMAP 3D 시각화 실패: {e}")

    rows = []
    for dim_name, mdict in emb_results.items():
        for mname, res in mdict.items():
            rows.append({
                "차원": dim_name, "Model": mname,
                "Accuracy": f"{res['Accuracy'][0]:.4f} ± {res['Accuracy'][1]:.4f}",
                "F1":       f"{res['F1'][0]:.4f} ± {res['F1'][1]:.4f}",
                "ROC-AUC":  f"{res['ROC-AUC'][0]:.4f} ± {res['ROC-AUC'][1]:.4f}",
            })
    emb_df = pd.DataFrame(rows)
    emb_path2 = os.path.join(OUTPUT_DIR, "embedding_results.csv")
    emb_df.to_csv(emb_path2, index=False)
    print(f"    임베딩 결과 저장: {emb_path2}")
    return emb_results

# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed",    action="store_true", help="임베딩 분류 생략")
    parser.add_argument("--embed-only",  action="store_true", help="임베딩 분류만 실행")
    args = parser.parse_args()

    df = load_data()

    if not args.embed_only:
        run_ml(df)

    if not args.no_embed:
        try:
            import umap
        except ImportError:
            print("    umap-learn 설치 중...")
            os.system("pip install umap-learn -q")
        run_embedding(df)

    print("\n[4/4] 완료! 결과 파일:")
    for f in ["ml_results.csv", "shap_feature_importance.png",
              "embedding_results.csv", "umap_3d.png",
              "klue_embeddings.npy"]:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            print(f"    {p}")

if __name__ == "__main__":
    main()
