"""Step 1 — 다이어트 표본 title 텍스트 피처(24종) + 썸네일-title KoCLIP 유사도 추출.

정은(urp_jungeun)의 add_title_features.py를 HEALTH 다이어트 표본에 맞게 튜닝.
- 입력: 00_sample/sampling.py의 load_diet_sample() (VLOG CSV + subcategory merge 대체)
- 라벨: 표본의 age_group 사용 (연령 비율 재계산 불필요, 1·2단계와 동일 표본)
- 원본에서 미사용이던 E5 임베딩 모델 로드는 제거
- 출력: outputs/diet_title_features.csv
- --eda 옵션: 추출된 피처로 34- vs 65+ 비교 EDA (그림은 outputs/figures/)

실행: conda urp_yena 환경에서
  python 01_add_title_features.py          # 피처 추출 (GPU 필요: KoCLIP)
  python 01_add_title_features.py --eda    # EDA만
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from kiwipiepy import Kiwi

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent / "00_sample"))
from sampling import load_diet_sample  # noqa: E402

OUT_DIR = STAGE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = str(OUT_DIR / "diet_title_features.csv")

# -------------------------------------------------------
# 패턴 / 사전 상수 (원본 그대로)
# -------------------------------------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001FA00-\U0001FA9F"
    "]+",
    flags=re.UNICODE,
)
LIST_NUMBER_PATTERN = re.compile(r"\d+\s*(가지|개|위|번|명|곳|편|장|권|단계|종)")

FIRST_PERSON = {"나", "저", "우리", "저희", "내", "제"}
SECOND_PERSON = {"너", "자네", "그대", "당신", "너희", "여러분"}
THIRD_PERSON = {"그", "그녀", "이분", "그분", "저분", "그들", "저들", "이들"}
ALL_PRONOUNS = FIRST_PERSON | SECOND_PERSON | THIRD_PERSON

SUPERLATIVES = {"가장", "제일", "최고", "최악", "최초", "최대", "최소", "최강", "역대", "유일", "단독"}
CURIOSITY = {"충격", "반전", "알고보니", "사실은", "알고 보니", "놀라운", "경악", "충격적", "반전있는", "결국", "드디어", "알려진"}

NOUN_TAGS = {"NNG", "NNP", "NNB", "NR", "NP"}
VERB_TAGS = {"VV", "VA", "VX", "VCP", "VCN"}
ADV_TAGS = {"MAG", "MAJ"}
SKIP_TAGS = {
    "JX", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ",
    "SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "W_EMOJI",
    "EP", "EC", "ETM", "ETN", "XPN", "SN",
}

KIWI_FEATURE_KEYS = [
    "noun_ratio", "verb_ratio", "adv_ratio", "nnp_ratio",
    "has_pronoun", "has_past_tense", "has_question",
    "is_sentence", "is_noun_phrase",
]

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

# -------------------------------------------------------
# 모델 (지연 로드)
# -------------------------------------------------------
_clip_model = _clip_processor = _kiwi = None


def load_models():
    global _clip_model, _clip_processor, _kiwi
    if _kiwi is not None:
        return
    print("[모델 로드] KoCLIP...")
    _clip_model = CLIPModel.from_pretrained("koclip/koclip-base-pt")
    _clip_processor = CLIPProcessor.from_pretrained("koclip/koclip-base-pt")
    _clip_model.eval()
    if torch.cuda.is_available():
        _clip_model = _clip_model.cuda()
    print("[모델 로드] Kiwi...")
    _kiwi = Kiwi()
    print("[모델 로드 완료]\n")


def preprocess_text(text):
    text = re.sub(r"(\S)(#)", r"\1 \2", str(text))
    return text.strip()


def english_ratio(text):
    text = str(text)
    eng = len(re.findall(r"[a-zA-Z]", text))
    total = len(re.sub(r"\s", "", text))
    return eng / total if total > 0 else 0


def get_clip_similarity(img_path, text):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = _clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _clip_model(**inputs)
        return round(outputs.logits_per_image.item(), 4)
    except Exception:
        return None


def extract_kiwi_features(text):
    try:
        result = _kiwi.analyze(text)[0][0]
        morphs = [(token.form, token.tag) for token in result]
        tags = [t for _, t in morphs]
        total = len(tags)
        if total == 0:
            raise ValueError("empty")

        last_content_tag = None
        for _, tag in reversed(morphs):
            if tag not in SKIP_TAGS:
                last_content_tag = tag
                break

        return {
            "noun_ratio": round(sum(1 for t in tags if t in NOUN_TAGS) / total, 4),
            "verb_ratio": round(sum(1 for t in tags if t in VERB_TAGS) / total, 4),
            "adv_ratio": round(sum(1 for t in tags if t in ADV_TAGS) / total, 4),
            "nnp_ratio": round(sum(1 for t in tags if t == "NNP") / total, 4),
            "has_pronoun": int(any(t == "NP" for t in tags)),
            "has_past_tense": int(any(t == "EP" and w in {"었", "았"} for w, t in morphs)),
            "has_question": int(any(t == "SF" and w == "?" for w, t in morphs)),
            "is_sentence": int(any(t == "EF" for t in tags)),
            "is_noun_phrase": int(last_content_tag in NOUN_TAGS),
        }
    except Exception:
        return {k: None for k in KIWI_FEATURE_KEYS}


def extract_features(title):
    text = preprocess_text(str(title))
    chars_no_space = re.sub(r"\s", "", text)
    words = text.split()

    try:
        result = _kiwi.analyze(text)[0][0]
        morph_words = [token.form for token in result]
        ttr = len(set(morph_words)) / len(morph_words) if morph_words else None
    except Exception:
        ttr = None

    result = {
        "char_count": len(chars_no_space),
        "word_count": len(words),
        "mean_word_length": round(sum(len(w) for w in words) / len(words), 4) if words else 0,
        "exclamation_count": sum(text.count(c) for c in "!！"),
        "ellipsis_count": text.count("…") + text.count("..."),
        "quotation_count": sum(text.count(c) for c in "\"'“”『』「」"),
        "bracket_count": sum(text.count(c) for c in "()[]{}【】《》"),
        "common_punct_count": sum(text.count(c) for c in ",.;:-"),
        "numbers_count": len(re.findall(r"\d+", text)),
        "emoji_count": len(EMOJI_PATTERN.findall(text)),
        "has_list_number": int(bool(LIST_NUMBER_PATTERN.search(text))),
        "personal_pronoun_count": sum(1 for w in words if w in ALL_PRONOUNS),
        "superlative_count": sum(1 for w in words if w in SUPERLATIVES),
        "curiosity_count": sum(1 for kw in CURIOSITY if kw in text),
        "ttr": round(ttr, 4) if ttr is not None else None,
    }
    result.update(extract_kiwi_features(text))
    return result


# -------------------------------------------------------
# 실행: 피처 추출
# -------------------------------------------------------
def add_features(output_path=OUTPUT_PATH, checkpoint_every=500):
    load_models()

    df = load_diet_sample()
    df["target"] = df["age_group"].map({"34-": 0, "65+": 1})
    print(f"[공용 표본 로드] {len(df)}개 (34-/65+ 각 {int(len(df)/2)}개)")

    eng_ratio = df["title"].apply(english_ratio)
    eng_dom = (eng_ratio >= 0.5).sum()
    df = df[eng_ratio < 0.5]
    print(f"  영어 50%+ 제거: {eng_dom}개 → 처리 대상: {len(df)}개\n")

    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path, low_memory=False)
        done_ids = set(done_df["video_id"].tolist())
        df = df[~df["video_id"].isin(done_ids)]
        print(f"[이어하기] 이미 처리됨: {len(done_ids)}개 → 남은 대상: {len(df)}개\n")

    processed = skipped = 0
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="피처 추출"):
        try:
            title = str(row["title"])
            img_path = row["resolved_path"]

            features = extract_features(title)
            features["clip_similarity"] = (
                get_clip_similarity(img_path, preprocess_text(title)) if os.path.exists(img_path) else None
            )
            records.append({**row.to_dict(), **features})
            processed += 1
        except Exception as e:
            tqdm.write(f"[오류] {row.get('title', '')} | {e}")
            skipped += 1

        if processed % checkpoint_every == 0 and processed > 0 and records:
            write_header = not os.path.exists(output_path)
            pd.DataFrame(records).to_csv(
                output_path, mode="a", index=False, header=write_header,
                encoding="utf-8-sig", quoting=csv.QUOTE_ALL,
            )
            tqdm.write(f"[체크포인트] {processed}개 저장 완료")
            records = []

    if records:
        write_header = not os.path.exists(output_path)
        pd.DataFrame(records).to_csv(
            output_path, mode="a", index=False, header=write_header,
            encoding="utf-8-sig", quoting=csv.QUOTE_ALL,
        )

    print(f"\n[완료] 성공: {processed}개 / 스킵: {skipped}개")
    print(f"[저장] {output_path}")


# -------------------------------------------------------
# EDA: 34- vs 65+ 비교 (원본 로직, 라벨만 age_group 기반)
# -------------------------------------------------------
def run_eda(output_path=OUTPUT_PATH):
    import koreanize_matplotlib  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(output_path)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    df_bin = df.dropna(subset=feature_cols + ["target"]).copy()
    df_bin["target"] = df_bin["target"].astype(int)

    print("[분포]")
    print(df_bin["age_group"].value_counts())
    print("샘플 수:", len(df_bin))

    young = df_bin[df_bin["target"] == 0]
    senior = df_bin[df_bin["target"] == 1]

    print("\n[기술통계 — 34-]")
    print(young[feature_cols].describe().round(3).to_string())
    print("\n[기술통계 — 65+]")
    print(senior[feature_cols].describe().round(3).to_string())

    missing = (df.reindex(columns=feature_cols).isna().mean() * 100).round(1).sort_values(ascending=False)
    print("\n[결측치 비율 (%)]")
    print(missing[missing > 0].to_string() if (missing > 0).any() else "없음")

    print("\n[그룹별 평균 비교 & t-test]")
    rows = []
    for col in feature_cols:
        y_vals, s_vals = young[col].dropna(), senior[col].dropna()
        t, p = stats.ttest_ind(y_vals, s_vals, equal_var=False)
        rows.append({
            "feature": col,
            "mean_~34": round(y_vals.mean(), 4),
            "mean_65~": round(s_vals.mean(), 4),
            "diff": round(s_vals.mean() - y_vals.mean(), 4),
            "t_stat": round(t, 3),
            "p_value": round(p, 4),
            "significant": "*" if p < 0.05 else "",
        })
    ttest_df = pd.DataFrame(rows).sort_values("p_value")
    print(ttest_df.to_string(index=False))

    corr = df_bin[feature_cols].corrwith(df_bin["target"]).sort_values(key=abs, ascending=False).round(4)
    print("\n[target 상관계수 (절댓값 순)]")
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(feature_cols))
    width = 0.4
    ax.bar([i - width / 2 for i in x], [young[c].mean() for c in feature_cols], width, label="34-", color="#378ADD", alpha=0.8)
    ax.bar([i + width / 2 for i in x], [senior[c].mean() for c in feature_cols], width, label="65+", color="#D85A30", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_title("피처별 그룹 평균 비교")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_group_mean.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#D85A30" if v > 0 else "#378ADD" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("피처 vs target 상관계수 (양수=65+ 경향)")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_correlation.png", dpi=150)
    plt.close(fig)

    sig_feats = ttest_df[ttest_df["p_value"] < 0.05]["feature"].tolist()[:6]
    if sig_feats:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, col in enumerate(sig_feats):
            ax = axes[i]
            ax.hist(young[col].dropna(), bins=30, alpha=0.6, label="34-", color="#378ADD", density=True)
            ax.hist(senior[col].dropna(), bins=30, alpha=0.6, label="65+", color="#D85A30", density=True)
            p_val = ttest_df[ttest_df["feature"] == col]["p_value"].values[0]
            ax.set_title(f"{col}  (p={p_val:.4f})", fontsize=10)
            ax.legend(fontsize=8)
        for j in range(len(sig_feats), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("유의미한 피처 분포 (p < 0.05)", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "eda_dist.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = df_bin[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("피처 간 상관 히트맵")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n[저장] {fig_dir}/eda_group_mean.png / eda_correlation.png / eda_dist.png / eda_heatmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", action="store_true", help="피처 추출 대신 EDA만 실행")
    args = parser.parse_args()

    if args.eda:
        run_eda()
    else:
        add_features()
