"""
4카테고리 통합 Grad-CAM 파이프라인 (DINOv2-base).

Stages:
  prepare | train | eval | gradcam | roi | all

Usage:
  python prepare_data.py
  python run_gradcam_pipeline.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

STAGE_DIR = Path(__file__).resolve().parent
OUT_DIR = STAGE_DIR / "outputs"
ROOT = STAGE_DIR.parent
YOLO_WEIGHTS = ROOT / "6_썸네일_attention_gradcam/yolov8n.pt"
SAMPLE_CSV = OUT_DIR / "all_sample.csv"
MODEL_CKPT = OUT_DIR / "dinov3_classifier.pt"
METRICS_JSON = OUT_DIR / "metrics.json"

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RANDOM_STATE = 42
EPOCHS = 5
BATCH_SIZE = 16
CLASS_NAMES = {0: "~34", 1: "65~"}
TOP_N = 50
ANALYSIS_N = 50


# ── Model / data ──────────────────────────────────────────────────────────────
class ThumbDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["resolved_path"]).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")
        pixel_values = x["pixel_values"].squeeze(0)
        y = int(row["target"])
        return pixel_values, y, row["resolved_path"], row["y_grouped"]


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    groups = [b[3] for b in batch]
    return pixel_values, labels, paths, groups


class DinoBinaryClassifier(nn.Module):
    """DINOv3 / DINOv2 공통 래퍼. encoder 경로 차이를 내부에서 처리."""

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",  # GradCAM gradient flow
        )
        hidden = self.backbone.config.hidden_size
        self.num_prefix = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def encoder_layers(self):
        # DINOv2: backbone.encoder.layer / DINOv3: backbone.model.layer
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
            return self.backbone.model.layer
        raise AttributeError("Cannot locate transformer layers on backbone")

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def get_device():
    if torch.cuda.is_available():
        # 여유 VRAM이 더 큰 GPU 선택
        free = []
        for i in range(torch.cuda.device_count()):
            try:
                free_b, _ = torch.cuda.mem_get_info(i)
                free.append((free_b, i))
            except Exception:
                free.append((0, i))
        free.sort(reverse=True)
        idx = free[0][1]
        torch.cuda.set_device(idx)
        return f"cuda:{idx}"
    return "cpu"


def load_sample_splits():
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Run prepare_data.py first → {SAMPLE_CSV}")
    df = pd.read_csv(SAMPLE_CSV)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["target"]
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        df,
    )


def check_model_access():
    try:
        AutoImageProcessor.from_pretrained(MODEL_NAME)
        # config only — full weights load in train
        AutoModel.from_pretrained(MODEL_NAME)
        return True
    except Exception as e:
        print("=" * 70)
        print("모델 가중치 로드 실패.")
        print(f"모델: {MODEL_NAME}")
        print(f"에러: {type(e).__name__}: {e}")
        print("=" * 70)
        return False


# ── Train / Eval ──────────────────────────────────────────────────────────────
def stage_train(device):
    train_df, test_df, _ = load_sample_splits()
    print(f"train={len(train_df)} test={len(test_df)}")
    print("train age:", train_df["y_grouped"].value_counts().to_dict())
    print("test age :", test_df["y_grouped"].value_counts().to_dict())
    print("train cat:", train_df["category"].value_counts().to_dict())

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = DinoBinaryClassifier(MODEL_NAME).to(device)
    print(f"backbone={MODEL_NAME} hidden={model.backbone.config.hidden_size} "
          f"num_prefix={model.num_prefix}")

    for p in model.backbone.parameters():
        p.requires_grad = False

    train_loader = DataLoader(
        ThumbDataset(train_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}", leave=True)
        for pixel_values, labels, _, _ in iterator:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        print(f"[epoch {epoch+1}/{EPOCHS}] loss={total/len(train_loader):.4f}", flush=True)

    # GradCAM용 마지막 블록 gradient 허용
    for p in model.encoder_layers()[-1].parameters():
        p.requires_grad = True

    torch.save(
        {
            "model_name": MODEL_NAME,
            "state_dict": model.state_dict(),
            "num_prefix": model.num_prefix,
            "train_n": len(train_df),
            "test_n": len(test_df),
        },
        MODEL_CKPT,
    )
    print("saved:", MODEL_CKPT)
    # persist splits for later stages
    train_df.to_csv(OUT_DIR / "train_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test_split.csv", index=False, encoding="utf-8-sig")
    return model, processor, train_df, test_df


def load_trained(device):
    if not MODEL_CKPT.exists():
        raise FileNotFoundError(MODEL_CKPT)
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)
    processor = AutoImageProcessor.from_pretrained(ckpt["model_name"])
    model = DinoBinaryClassifier(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.encoder_layers()[-1].parameters():
        p.requires_grad = True
    train_df = pd.read_csv(OUT_DIR / "train_split.csv")
    test_df = pd.read_csv(OUT_DIR / "test_split.csv")
    return model, processor, train_df, test_df


def stage_eval(model, processor, test_df, device):
    loader = DataLoader(
        ThumbDataset(test_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for pixel_values, labels, _, _ in loader:
            pred = torch.argmax(model(pixel_values.to(device)), dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(pred)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["~34", "65~"], output_dict=True)
    print("accuracy        :", acc)
    print("balanced_accuracy:", bacc)
    print(classification_report(y_true, y_pred, target_names=["~34", "65~"]))

    # per-category
    test_eval = test_df.copy().reset_index(drop=True)
    test_eval["pred"] = y_pred
    per_cat = {}
    for cat, g in test_eval.groupby("category"):
        a = accuracy_score(g["target"], g["pred"])
        per_cat[cat] = float(a)
        print(f"  [{cat}] acc={a:.4f} n={len(g)}")

    metrics = {
        "model": MODEL_NAME,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "report": report,
        "per_category_accuracy": per_cat,
        "n_test": len(test_df),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", METRICS_JSON)
    return metrics


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def make_cam(model):
    from pytorch_grad_cam import GradCAM

    num_prefix = model.num_prefix

    def reshape_transform(tensor):
        # [B, 1+R+N, C] → drop CLS(+register) → [B, C, H, W]
        tensor = tensor[:, num_prefix:, :]
        n_patches = tensor.size(1)
        h = w = int(n_patches**0.5)
        assert h * w == n_patches, f"Patch count ({n_patches}) is not square (prefix={num_prefix})"
        tensor = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
        return tensor.permute(0, 3, 1, 2).contiguous()

    target_layers = [model.encoder_layers()[-1].norm1]
    return GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


def run_gradcam_on_image(model, processor, cam, image_path, device, save_path=None):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    img_orig = Image.open(image_path).convert("RGB")
    inputs = processor(images=img_orig, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        probs = torch.softmax(model(pixel_values), dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))

    _, _, H, W = pixel_values.shape
    img_resized = img_orig.resize((W, H))
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0

    cam_young = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(0)])[0]
    cam_old = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(1)])[0]
    vis_young = show_cam_on_image(rgb_img, cam_young, use_rgb=True)
    vis_old = show_cam_on_image(rgb_img, cam_old, use_rgb=True)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title(
        f"Original\npred={CLASS_NAMES[pred_class]}  "
        f"({probs[0]*100:.1f}% ~34 / {probs[1]*100:.1f}% 65~)"
    )
    axes[0].axis("off")
    axes[1].imshow(vis_young)
    axes[1].set_title("GradCAM -> ~34")
    axes[1].axis("off")
    im0 = axes[2].imshow(cam_young, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Heatmap (~34)")
    axes[2].axis("off")
    plt.colorbar(im0, ax=axes[2], fraction=0.046)
    axes[3].imshow(vis_old)
    axes[3].set_title("GradCAM -> 65~")
    axes[3].axis("off")
    im1 = axes[4].imshow(cam_old, cmap="jet", vmin=0, vmax=1)
    axes[4].set_title("Heatmap (65~)")
    axes[4].axis("off")
    plt.colorbar(im1, ax=axes[4], fraction=0.046)
    plt.suptitle(os.path.basename(image_path), fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
    plt.close(fig)


def show_gradcam_grid(model, processor, cam, df_sub, device, max_images=10, save_path="gradcam_grid.png"):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    df_sub = df_sub.head(max_images).reset_index(drop=True)
    n = len(df_sub)
    fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    model.eval()
    for i, (_, row) in enumerate(df_sub.iterrows()):
        img_orig = Image.open(row["resolved_path"]).convert("RGB")
        inputs = processor(images=img_orig, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        _, _, H, W = pixel_values.shape
        img_resized = img_orig.resize((W, H))
        rgb_img = np.array(img_resized).astype(np.float32) / 255.0
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values), dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        cam_young = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(0)])[0]
        cam_old = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(1)])[0]
        vis_young = show_cam_on_image(rgb_img, cam_young, use_rgb=True)
        vis_old = show_cam_on_image(rgb_img, cam_old, use_rgb=True)
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(
            f"pred={CLASS_NAMES[pred_class]} ({probs[0]*100:.0f}/{probs[1]*100:.0f})\n"
            f"{row.get('category','')} | {str(row.get('channel_name',''))[:20]}",
            fontsize=8,
        )
        axes[i, 0].axis("off")
        axes[i, 1].imshow(vis_young)
        axes[i, 1].set_title("GradCAM -> ~34", fontsize=8)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(cam_young, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title("Heatmap ~34", fontsize=8)
        axes[i, 2].axis("off")
        axes[i, 3].imshow(vis_old)
        axes[i, 3].set_title("GradCAM -> 65~", fontsize=8)
        axes[i, 3].axis("off")
        axes[i, 4].imshow(cam_old, cmap="jet", vmin=0, vmax=1)
        axes[i, 4].set_title("Heatmap 65~", fontsize=8)
        axes[i, 4].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print("Saved:", save_path)
    plt.close(fig)


def dedup_by_channel(df_sorted, top_n=50):
    if "channel_id" not in df_sorted.columns:
        return df_sorted.head(top_n)
    seen, selected = set(), []
    for _, row in df_sorted.iterrows():
        ch = row["channel_id"]
        if ch not in seen:
            seen.add(ch)
            selected.append(row)
        if len(selected) == top_n:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def stage_gradcam(model, processor, test_df, device):
    cam = make_cam(model)
    loader = DataLoader(
        ThumbDataset(test_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for pixel_values, labels, paths, _ in loader:
            probs = torch.softmax(model(pixel_values.to(device)), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))

    result_df = test_df.copy().reset_index(drop=True)
    result_df["pred"] = all_preds
    result_df["prob_65"] = [p[1] for p in all_probs]
    result_df["correct"] = result_df["target"] == result_df["pred"]
    result_df.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    run_gradcam_on_image(
        model, processor, cam, result_df.iloc[0]["resolved_path"], device,
        save_path=str(OUT_DIR / "gradcam_single.png"),
    )

    correct_65 = dedup_by_channel(
        result_df[(result_df.correct) & (result_df.target == 1)].sort_values("prob_65", ascending=False),
        TOP_N,
    )
    correct_34 = dedup_by_channel(
        result_df[(result_df.correct) & (result_df.target == 0)].sort_values("prob_65", ascending=True),
        TOP_N,
    )
    incorrect = result_df[~result_df.correct].copy()
    if len(incorrect):
        incorrect["conf"] = incorrect.apply(
            lambda r: r["prob_65"] if r["pred"] == 1 else (1 - r["prob_65"]), axis=1
        )
        incorrect_s = dedup_by_channel(incorrect.sort_values("conf", ascending=False), TOP_N)
    else:
        incorrect_s = incorrect

    show_gradcam_grid(model, processor, cam, correct_65, device, TOP_N, OUT_DIR / "gradcam_correct_65_top50.png")
    show_gradcam_grid(model, processor, cam, correct_34, device, TOP_N, OUT_DIR / "gradcam_correct_34_top50.png")
    if len(incorrect_s):
        show_gradcam_grid(model, processor, cam, incorrect_s, device, TOP_N, OUT_DIR / "gradcam_incorrect_top50.png")

    cols = [c for c in ["resolved_path", "category", "channel_name", "channel_id", "title",
                        "video_id", "target", "pred", "prob_65", "correct"] if c in correct_65.columns]
    correct_65[cols].to_csv(OUT_DIR / "samples_correct_65_top50.csv", index=False, encoding="utf-8-sig")
    correct_34[cols].to_csv(OUT_DIR / "samples_correct_34_top50.csv", index=False, encoding="utf-8-sig")
    if len(incorrect_s):
        incorrect_s[cols].to_csv(OUT_DIR / "samples_incorrect_top50.csv", index=False, encoding="utf-8-sig")

    return cam, correct_65, correct_34


# ── ROI analysis ──────────────────────────────────────────────────────────────
def stage_roi(model, processor, cam, correct_65, correct_34, device):
    import easyocr
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from ultralytics import YOLO

    print("Initializing EasyOCR...")
    ocr_reader = easyocr.Reader(["ko", "en"], gpu=str(device).startswith("cuda"), verbose=False)
    print("Initializing YOLOv8...", YOLO_WEIGHTS)
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    def get_prob(pixel_values):
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(pixel_values), dim=1)[0].cpu().numpy()

    def get_pixel_values(img_pil):
        inputs = processor(images=img_pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        _, _, H, W = pv.shape
        img_resized = img_pil.resize((W, H))
        rgb_np = np.array(img_resized).astype(np.float32) / 255.0
        return pv, rgb_np, H, W

    def get_cam(pixel_values, target_class):
        return cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(target_class)])[0]

    def build_roi_masks(image_path, H, W):
        img_orig = Image.open(image_path).convert("RGB")
        orig_W, orig_H = img_orig.size
        img_np = np.array(img_orig)
        mask_text = np.zeros((orig_H, orig_W), dtype=bool)
        mask_person = np.zeros((orig_H, orig_W), dtype=bool)
        for bbox, text, conf in ocr_reader.readtext(
            img_np, detail=1, paragraph=False,
            min_size=10, text_threshold=0.4, low_text=0.3,
            link_threshold=0.3, width_ths=0.8,
            contrast_ths=0.05, adjust_contrast=0.7,
        ):
            if conf < 0.3:
                continue
            pts = np.array(bbox, dtype=np.int32)
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(orig_W, x1), min(orig_H, y1)
            mask_text[y0:y1, x0:x1] = True
        for box in yolo_model(img_np, verbose=False)[0].boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.3:
                continue
            x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(orig_W, x1), min(orig_H, y1)
            mask_person[y0:y1, x0:x1] = True
        mask_bg = ~(mask_text | mask_person)

        def resize_mask(m):
            return np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            ) > 127

        return {
            "text": resize_mask(mask_text),
            "person": resize_mask(mask_person),
            "background": resize_mask(mask_bg),
        }

    cache_file = OUT_DIR / "roi_masks_cache.pkl"
    if cache_file.exists():
        print("Loading ROI cache:", cache_file)
        with open(cache_file, "rb") as f:
            roi_cache = pickle.load(f)
    else:
        roi_cache = {}
        paths = list(dict.fromkeys(
            correct_65.head(ANALYSIS_N)["resolved_path"].tolist()
            + correct_34.head(ANALYSIS_N)["resolved_path"].tolist()
        ))
        print(f"Building ROI masks for {len(paths)} images...")
        for i, path in enumerate(paths):
            try:
                # processor size may not be 224 for dinov3 — probe once
                img = Image.open(path).convert("RGB")
                pv, _, H, W = get_pixel_values(img)
                roi_cache[(path, H, W)] = build_roi_masks(path, H, W)
            except Exception as e:
                print(f"  WARNING {Path(path).name}: {e}")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(paths)}")
        with open(cache_file, "wb") as f:
            pickle.dump(roi_cache, f)
        print("Cache saved:", cache_file)

    def get_roi_masks_cached(image_path, H, W):
        key = (image_path, H, W)
        if key not in roi_cache:
            roi_cache[key] = build_roi_masks(image_path, H, W)
        return roi_cache[key]

    def roi_deletion_insertion(image_path):
        img_orig = Image.open(image_path).convert("RGB")
        pv, rgb_np, H, W = get_pixel_values(img_orig)
        orig_probs = get_prob(pv)
        pred_class = int(np.argmax(orig_probs))
        orig_prob = float(orig_probs[pred_class])
        blurred = np.array(img_orig.resize((W, H)).filter(ImageFilter.GaussianBlur(radius=11))).astype(np.float32) / 255.0
        mean_val = rgb_np.mean(axis=(0, 1), keepdims=True)
        roi_masks = get_roi_masks_cached(image_path, H, W)
        if not roi_masks["text"].any() and not roi_masks["person"].any():
            roi_masks["background"] = np.ones((H, W), dtype=bool)
        results = {}
        for roi_name, mask in roi_masks.items():
            m = mask[..., None].astype(np.float32)
            del_img = rgb_np * (1 - m) + mean_val * m
            del_pv, _, _, _ = get_pixel_values(Image.fromarray((del_img * 255).astype(np.uint8)))
            del_prob = float(get_prob(del_pv)[pred_class])
            ins_img = rgb_np * m + blurred * (1 - m)
            ins_pv, _, _, _ = get_pixel_values(Image.fromarray((ins_img * 255).astype(np.uint8)))
            ins_prob = float(get_prob(ins_pv)[pred_class])
            results[roi_name] = {"del": del_prob, "ins": ins_prob}
        return results, pred_class, orig_prob, roi_masks

    def batch_roi_analysis(sample_df, label, save_path):
        sample_df = sample_df.head(ANALYSIS_N).reset_index(drop=True)
        roi_del, roi_ins, orig_list, roi_pixel_ratios = {}, {}, {}, {}
        for _, row in sample_df.iterrows():
            try:
                res, _, orig_p, roi_masks = roi_deletion_insertion(row["resolved_path"])
                orig_list.setdefault("vals", []).append(orig_p)
                for rname, vals in res.items():
                    roi_del.setdefault(rname, []).append(vals["del"])
                    roi_ins.setdefault(rname, []).append(vals["ins"])
                    roi_pixel_ratios.setdefault(rname, []).append(float(roi_masks[rname].mean()))
            except Exception as e:
                print(f"  WARNING skip: {e}")
        if not roi_del:
            print("No valid samples")
            return
        roi_names = list(roi_del.keys())
        mean_orig = np.mean(orig_list["vals"])
        mean_dels = [np.mean(roi_del[r]) for r in roi_names]
        mean_ins_v = [np.mean(roi_ins[r]) for r in roi_names]
        mean_ratios = [np.mean(roi_pixel_ratios[r]) for r in roi_names]
        eps = 1e-6
        norm_del = [(mean_orig - mean_dels[i]) / (mean_ratios[i] + eps) for i in range(len(roi_names))]
        norm_ins = [mean_ins_v[i] / (mean_ratios[i] + eps) for i in range(len(roi_names))]
        x = np.arange(len(roi_names))
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, norm_vals, raw_vals, direction, ylabel in [
            (axes[0], norm_del, mean_dels,
             f"Deletion Importance\n[{label}]",
             "Normalized [(orig - del) / area]"),
            (axes[1], norm_ins, mean_ins_v,
             f"Insertion Importance\n[{label}]",
             "Normalized [ins / area]"),
        ]:
            bars = ax.bar(x, norm_vals, 0.4, color=colors[: len(roi_names)], alpha=0.85)
            ax.set_title(direction, fontsize=10, pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{r}\n(area {mean_ratios[i]*100:.1f}%)" for i, r in enumerate(roi_names)],
                fontsize=10,
            )
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            max_val = max(norm_vals) if norm_vals else 1
            ax.set_ylim(0, max(max_val, 0.01) * 1.25)
            for bar, nv, rv in zip(bars, norm_vals, raw_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(max_val, 0.01) * 0.02,
                    f"{nv:.2f}\n(raw:{rv:.3f})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
        plt.suptitle(
            f"ROI Normalized Del/Ins [{label}] n={len(orig_list['vals'])} | baseline={mean_orig:.3f}",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved:", save_path)
        plt.close(fig)
        print(f"  [{label}] baseline={mean_orig:.3f}")
        for i, rname in enumerate(roi_names):
            print(
                f"  {rname:<12} area={mean_ratios[i]*100:5.1f}% "
                f"del={mean_dels[i]:.3f} ins={mean_ins_v[i]:.3f} "
                f"delN={norm_del[i]:.2f} insN={norm_ins[i]:.2f}"
            )

    def batch_deletion_insertion(sample_df, label, save_path, n_steps=10):
        sample_df = sample_df.head(ANALYSIS_N).reset_index(drop=True)
        all_del, all_ins, thrs = [], [], None
        for _, row in sample_df.iterrows():
            try:
                img_orig = Image.open(row["resolved_path"]).convert("RGB")
                pv, rgb_np, H, W = get_pixel_values(img_orig)
                orig_probs = get_prob(pv)
                pred_class = int(np.argmax(orig_probs))
                grayscale_cam = get_cam(pv, pred_class)
                thrs = np.linspace(0, 1, n_steps + 1)[1:]
                mean_val = rgb_np.mean(axis=(0, 1), keepdims=True)
                blurred = np.array(
                    img_orig.resize((W, H)).filter(ImageFilter.GaussianBlur(radius=11))
                ).astype(np.float32) / 255.0
                dels, ins = [], []
                for thr in thrs:
                    mask = (grayscale_cam >= np.quantile(grayscale_cam, 1 - thr)).astype(np.float32)
                    del_img = rgb_np * (1 - mask[..., None]) + mean_val * mask[..., None]
                    del_pv, _, _, _ = get_pixel_values(Image.fromarray((del_img * 255).astype(np.uint8)))
                    dels.append(float(get_prob(del_pv)[pred_class]))
                    ins_img = rgb_np * mask[..., None] + blurred * (1 - mask[..., None])
                    ins_pv, _, _, _ = get_pixel_values(Image.fromarray((ins_img * 255).astype(np.uint8)))
                    ins.append(float(get_prob(ins_pv)[pred_class]))
                all_del.append(dels)
                all_ins.append(ins)
            except Exception as e:
                print(f"  WARNING skip: {e}")
        if not all_del:
            return
        all_del = np.array(all_del)
        all_ins = np.array(all_ins)
        mean_del, mean_ins = all_del.mean(0), all_ins.mean(0)
        # numpy<2: trapz / numpy>=2: trapezoid
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        auc_del = float(_trapz(mean_del, thrs))
        auc_ins = float(_trapz(mean_ins, thrs))
        pct_labels = [f"{int(t*100)}%" for t in thrs]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, curves, mean_c, c1, c2, title, xlabel in [
            (axes[0], all_del, mean_del, "salmon", "red",
             f"Deletion — {label}\n(lower AUC better)", "Fraction deleted"),
            (axes[1], all_ins, mean_ins, "skyblue", "blue",
             f"Insertion — {label}\n(higher AUC better)", "Fraction revealed"),
        ]:
            for curve in curves:
                ax.plot(thrs, curve, color=c1, alpha=0.3, linewidth=0.8)
            auc = float(_trapz(mean_c, thrs))
            ax.plot(thrs, mean_c, color=c2, linewidth=2.5, label=f"Mean (AUC={auc:.3f})")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Prediction probability")
            ax.set_xticks(thrs)
            ax.set_xticklabels(pct_labels, rotation=45)
            ax.legend()
            ax.grid(alpha=0.3)
        plt.suptitle(f"Del/Ins [{label}] n={len(all_del)} | DelAUC={auc_del:.3f} InsAUC={auc_ins:.3f}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path} | DelAUC={auc_del:.4f} InsAUC={auc_ins:.4f}")
        plt.close(fig)

    def show_roi_detection(sample_df, n_show, save_path):
        sample_df = sample_df.head(n_show).reset_index(drop=True)
        fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]
        for j, title in enumerate(["Original", "Text", "Person", "Background", "Composite"]):
            axes[0, j].set_title(title, fontsize=10, fontweight="bold")
        for i, (_, row) in enumerate(sample_df.iterrows()):
            img_orig = Image.open(row["resolved_path"]).convert("RGB")
            pv, rgb_np, H, W = get_pixel_values(img_orig)
            masks = get_roi_masks_cached(row["resolved_path"], H, W)
            axes[i, 0].imshow((rgb_np * 255).astype(np.uint8))
            axes[i, 0].set_title(f"{row.get('category','')} | {str(row.get('channel_name',''))[:18]}", fontsize=7)
            axes[i, 0].axis("off")
            for j, (name, color) in enumerate(
                [("text", [255, 80, 80]), ("person", [80, 120, 255]), ("background", [80, 200, 80])], start=1
            ):
                m = masks[name]
                vis = (rgb_np * 255).astype(np.uint8).copy()
                vis[m] = (vis[m] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
                vis[~m] = (vis[~m] * 0.3).astype(np.uint8)
                axes[i, j].imshow(vis)
                axes[i, j].set_title(f"{name} {m.mean()*100:.1f}%", fontsize=8)
                axes[i, j].axis("off")
            composite = (rgb_np * 255).astype(np.uint8).copy()
            composite[masks["text"]] = (composite[masks["text"]] * 0.45 + np.array([255, 80, 80]) * 0.55).astype(np.uint8)
            composite[masks["person"]] = (composite[masks["person"]] * 0.45 + np.array([80, 120, 255]) * 0.55).astype(np.uint8)
            composite[masks["background"]] = (composite[masks["background"]] * 0.75).astype(np.uint8)
            axes[i, 4].imshow(composite)
            axes[i, 4].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print("Saved:", save_path)
        plt.close(fig)

    def masking_visualization_grid(sample_df, save_path, thresholds=(0.3, 0.5, 0.7)):
        sample_df = sample_df.reset_index(drop=True)
        n = len(sample_df)
        ncols = 2 + len(thresholds)
        fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        ROI_COLORS = {"text": [255, 80, 80], "person": [80, 120, 255], "background": [80, 200, 80]}
        for i, (_, row) in enumerate(sample_df.iterrows()):
            img_orig = Image.open(row["resolved_path"]).convert("RGB")
            pv, rgb_np, H, W = get_pixel_values(img_orig)
            orig_probs = get_prob(pv)
            pred_class = int(np.argmax(orig_probs))
            grayscale_cam = get_cam(pv, pred_class)
            roi_masks = get_roi_masks_cached(row["resolved_path"], H, W)
            overlay = (rgb_np * 255).astype(np.uint8).copy()
            for rname, rmask in roi_masks.items():
                color = np.array(ROI_COLORS[rname], dtype=np.uint8)
                overlay[rmask] = (overlay[rmask] * 0.6 + color * 0.4).astype(np.uint8)
            axes[i, 0].imshow(overlay)
            axes[i, 0].set_title(
                f"pred={CLASS_NAMES[pred_class]} ({orig_probs[0]*100:.0f}/{orig_probs[1]*100:.0f})\n"
                f"{row.get('category','')} | {str(row.get('channel_name',''))[:20]}",
                fontsize=7,
            )
            axes[i, 0].axis("off")
            axes[i, 1].imshow(grayscale_cam, cmap="jet", vmin=0, vmax=1)
            axes[i, 1].set_title("Heatmap", fontsize=8)
            axes[i, 1].axis("off")
            for j, thr in enumerate(thresholds):
                mask = grayscale_cam >= thr
                masked = rgb_np.copy()
                masked[~mask] = 0.0
                axes[i, 2 + j].imshow((masked * 255).astype(np.uint8))
                axes[i, 2 + j].set_title(f"thr>={thr:.1f} | {mask.mean()*100:.0f}%", fontsize=8)
                axes[i, 2 + j].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print("Saved:", save_path)
        plt.close(fig)

    print("[1] Del/Ins curves")
    batch_deletion_insertion(correct_65, "65~ correct", OUT_DIR / "del_ins_65.png")
    batch_deletion_insertion(correct_34, "~34 correct", OUT_DIR / "del_ins_34.png")
    print("[2] ROI-wise analysis")
    batch_roi_analysis(correct_65, "65~ correct", OUT_DIR / "roi_analysis_65.png")
    batch_roi_analysis(correct_34, "~34 correct", OUT_DIR / "roi_analysis_34.png")
    print("[3] ROI detection check")
    show_roi_detection(correct_65, 10, OUT_DIR / "roi_check_65.png")
    show_roi_detection(correct_34, 10, OUT_DIR / "roi_check_34.png")
    print("[4] Masking grids")
    masking_visualization_grid(correct_65.iloc[:25], OUT_DIR / "masking_65_grid_1.png")
    masking_visualization_grid(correct_65.iloc[25:50], OUT_DIR / "masking_65_grid_2.png")
    masking_visualization_grid(correct_34.iloc[:25], OUT_DIR / "masking_34_grid_1.png")
    masking_visualization_grid(correct_34.iloc[25:50], OUT_DIR / "masking_34_grid_2.png")
    print("ROI stage complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "gradcam", "roi", "all"],
        default="all",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT_DIR)

    if not SAMPLE_CSV.exists():
        print("Preparing merged sample...")
        from prepare_data import main as prep

        prep()

    print(f"Checking model access: {MODEL_NAME}")
    if not check_model_access():
        sys.exit(2)

    device = get_device()
    print("device:", device, "| model:", MODEL_NAME)

    if args.stage in ("train", "all"):
        model, processor, train_df, test_df = stage_train(device)
    else:
        model, processor, train_df, test_df = load_trained(device)

    if args.stage in ("eval", "all"):
        stage_eval(model, processor, test_df, device)

    if args.stage in ("gradcam", "roi", "all"):
        cam, correct_65, correct_34 = stage_gradcam(model, processor, test_df, device)
    else:
        cam = correct_65 = correct_34 = None

    if args.stage in ("roi", "all"):
        if cam is None:
            cam = make_cam(model)
            correct_65 = pd.read_csv(OUT_DIR / "samples_correct_65_top50.csv")
            correct_34 = pd.read_csv(OUT_DIR / "samples_correct_34_top50.csv")
        stage_roi(model, processor, cam, correct_65, correct_34, device)

    print("\nAll done. outputs →", OUT_DIR)


if __name__ == "__main__":
    main()
