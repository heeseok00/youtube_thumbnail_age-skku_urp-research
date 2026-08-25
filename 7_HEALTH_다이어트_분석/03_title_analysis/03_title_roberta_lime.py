"""Step 3 — klue/roberta-base 파인튜닝 + LIME 형태소 기여도 분석.

정은의 title_roberta_lime.py를 다이어트 표본에 맞게 튜닝.
- 입력: 00_sample/sampling.py의 load_diet_sample() (VLOG CSV + subcategory merge 대체)
- 라벨: 표본의 age_group 사용
- 전처리(한국어 비율 50%+, 중복 title 제거, 5자 미만 제거)와
  파인튜닝/LIME/시각화 로직은 원본 그대로
- 출력: outputs/figures/lime_morpheme_top_words.png + lime_word_weights.csv

실행: conda urp_yena 환경에서 (GPU 필요)
  python 03_title_roberta_lime.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kiwipiepy import Kiwi
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import re

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent / "00_sample"))
from sampling import load_diet_sample  # noqa: E402

OUT_DIR = STAGE_DIR / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "klue/roberta-base"
MAX_LEN = 64
EPOCHS = 5
LIME_SAMPLE_SIZE = 500
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kiwi = Kiwi()


def clean_title(text):
    text = re.sub(r"[ㅣ|]", " ", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def morpheme_tokenize(text):
    tokens = kiwi.tokenize(text)
    return " ".join(t.form for t in tokens)


def korean_char_ratio(text):
    if not text:
        return 0
    korean_chars = sum(1 for c in text if "가" <= c <= "힣")
    return korean_chars / len(text)


def load_data():
    df_bin = load_diet_sample()
    df_bin["target"] = df_bin["age_group"].map({"34-": 0, "65+": 1}).astype(int)
    print(f"[공용 표본 로드] {len(df_bin)}개")

    df_bin["korean_ratio"] = df_bin["title"].fillna("").apply(korean_char_ratio)
    df_bin = df_bin[df_bin["korean_ratio"] >= 0.5]
    df_bin = df_bin.drop_duplicates(subset="title")
    df_bin = df_bin[df_bin["title"].str.len() >= 5].reset_index(drop=True)

    print(f"[전처리 후] {len(df_bin)}개 | 34-: {(df_bin['target'] == 0).sum()}개 | 65+: {(df_bin['target'] == 1).sum()}개")

    texts_morphed = df_bin["title"].fillna("").apply(clean_title).apply(morpheme_tokenize).tolist()
    labels = df_bin["target"].values
    return texts_morphed, labels


class TitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    print(classification_report(all_labels, all_preds, target_names=["~34", "65~"]))
    print(f"AUC: {auc:.4f}")
    return auc


def finetune(texts_morphed, labels):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    X_train, X_val, y_train, y_val = train_test_split(
        texts_morphed, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(TitleDataset(X_train, y_train, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(TitleDataset(X_val, y_val, tokenizer), batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        print(f"\nEpoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, val_loader)

    return model, tokenizer, X_val, y_val


def run_lime(model, tokenizer, X_val, y_val):
    def predict_proba(texts):
        model.eval()
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.softmax(outputs.logits, dim=1).cpu().numpy()

    explainer = LimeTextExplainer(class_names=["~34", "65~"])

    sample_text = X_val[0]
    exp = explainer.explain_instance(sample_text, predict_proba, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
    print(f"\n제목(형태소): {sample_text}")
    print(f"정답: {'65세 이상' if y_val[0] == 1 else '34세 이하'}")
    print("LIME 기여 형태소:")
    for word, weight in exp.as_list():
        direction = "65세이상↑" if weight > 0 else "34세이하↑"
        print(f"  {word}: {weight:.4f} ({direction})")

    word_weights = defaultdict(list)
    sample_size = min(LIME_SAMPLE_SIZE, len(X_val))
    for i in tqdm(range(sample_size), desc="LIME 집계"):
        exp = explainer.explain_instance(X_val[i], predict_proba, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
        for word, weight in exp.as_list():
            word_weights[word].append(weight)

    return word_weights


def is_not_proper_noun(word):
    tokens = kiwi.tokenize(word)
    return not all(t.tag == "NNP" for t in tokens) and len(word) >= 2


def extract_top_words(word_weights, top_n=20, min_count=5):
    stopwords = {"이", "그", "저", "것", "수", "등", "및", "또", "더", "안", "못", "vs"}
    mean_weights = {
        w: np.mean(v)
        for w, v in word_weights.items()
        if len(v) >= min_count and w not in stopwords and is_not_proper_noun(w)
    }

    young_top = sorted(mean_weights.items(), key=lambda x: x[1])[:top_n]
    old_top = sorted(mean_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return young_top, old_top, mean_weights


def plot_top_words(young_top, old_top, out_path=None):
    out_path = out_path or (OUT_DIR / "lime_morpheme_top_words.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    words_y = [w for w, _ in young_top]
    vals_y = [v for _, v in young_top]
    axes[0].barh(words_y[::-1], vals_y[::-1], color="#3B8BD4")
    axes[0].set_title(f"34세 이하 예측 기여 형태소 Top {len(young_top)}")
    axes[0].set_xlabel("LIME weight")
    axes[0].axvline(0, color="gray", linewidth=0.8)

    words_o = [w for w, _ in old_top]
    vals_o = [v for _, v in old_top]
    axes[1].barh(words_o[::-1], vals_o[::-1], color="#E8593C")
    axes[1].set_title(f"65세 이상 예측 기여 형태소 Top {len(old_top)}")
    axes[1].set_xlabel("LIME weight")
    axes[1].axvline(0, color="gray", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    texts_morphed, labels = load_data()
    model, tokenizer, X_val, y_val = finetune(texts_morphed, labels)
    word_weights = run_lime(model, tokenizer, X_val, y_val)
    young_top, old_top, mean_weights = extract_top_words(word_weights)

    print("\n[34세 이하 기여 형태소 Top]")
    for w, v in young_top:
        print(f"  {w}: {v:.4f}")
    print("\n[65세 이상 기여 형태소 Top]")
    for w, v in old_top:
        print(f"  {w}: {v:.4f}")

    # 전체 형태소 weight 집계도 CSV로 보존 (재해석용)
    pd.DataFrame(
        [{"word": w, "mean_weight": np.mean(v), "count": len(v)} for w, v in word_weights.items()]
    ).sort_values("mean_weight").to_csv(
        OUT_DIR / "lime_word_weights.csv", index=False, encoding="utf-8-sig"
    )

    plot_top_words(young_top, old_top)
