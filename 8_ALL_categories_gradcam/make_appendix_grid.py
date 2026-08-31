"""부록용 Grad-CAM 격자 생성 (그룹당 12장 = 카테고리당 3장).

50장 격자는 지면에 넣기에 너무 크므로, 본문 Figure 1과 같은 표본에서
각 집단의 전형적인 양상을 가장 뚜렷하게 보여주는 예시를 카테고리당 3장씩
뽑아 원본/예측 클래스 오버레이 쌍으로 배치한다.

선택 기준
  65+ : 히트맵 에너지의 text + person 비율이 가장 높은 이미지
  ~34 : 히트맵이 가장 넓게 퍼진 이미지 (에너지 50%를 담는 데 필요한 면적 비율)

Usage:
  python make_appendix_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_gradcam_pipeline import OUT_DIR, get_device, load_trained, make_cam

# CSV의 category 값 → 논문 표기
CATEGORY_LABEL = {
    "EDU": "EDU",
    "HEALTH": "HEALTH",
    "MEDITATION": "LIFESTYLE",
    "SOCIETY": "SOCIETY",
}
CATEGORY_ORDER = ["EDU", "HEALTH", "MEDITATION", "SOCIETY"]
GROUP_KEY = {"65": "65~", "34": "~34"}
PER_CATEGORY = 3


def model_device(model):
    return next(model.parameters()).device


def compute_cam(model, processor, cam, image_path, target_class):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(
        model_device(model)
    )
    heat = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(target_class)])[0]
    _, _, H, W = pixel_values.shape
    return img.resize((W, H)), heat


def energy_half_area(heat):
    """히트맵 에너지의 50%를 담는 데 필요한 면적 비율. 클수록 넓게 퍼져 있다."""
    flat = np.sort(heat.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(flat) / total
    return float((np.searchsorted(cum, 0.5) + 1) / flat.size)


def select_examples(group, model, processor, cam):
    df = pd.read_csv(OUT_DIR / "cam_roi_share_per_image.csv")
    df = df[df["group"] == GROUP_KEY[group]].copy()

    if group == "65":
        df["score"] = df["share_text"].fillna(0) + df["share_person"].fillna(0)
    else:
        df["score"] = [
            energy_half_area(compute_cam(model, processor, cam, p, 0)[1])
            for p in df["resolved_path"]
        ]

    picked = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat].sort_values("score", ascending=False)
        if len(sub) < PER_CATEGORY:
            raise ValueError(f"{group}/{cat}: {len(sub)} available, need {PER_CATEGORY}")
        picked.append(sub.head(PER_CATEGORY))
    return pd.concat(picked, ignore_index=True)


def render(model, processor, cam, df, group, save_path):
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_class = 1 if group == "65" else 0
    ncols = PER_CATEGORY * 2
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), ncols, figsize=(2.6 * ncols, 3.0 * len(CATEGORY_ORDER))
    )

    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, PER_CATEGORY)
        img_resized, heat = compute_cam(model, processor, cam, row["resolved_path"], target_class)
        rgb = np.array(img_resized).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)

        ax_o, ax_c = axes[r, c * 2], axes[r, c * 2 + 1]
        ax_o.imshow(img_resized)
        ax_c.imshow(overlay)
        for ax in (ax_o, ax_c):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if c == 0:
            ax_o.set_ylabel(CATEGORY_LABEL[row["category"]], fontsize=11, labelpad=8)
        ax_o.set_title("original", fontsize=8)
        ax_c.set_title("Grad-CAM", fontsize=8)

    label = "65+" if group == "65" else "~34"
    fig.suptitle(f"Grad-CAM for correctly classified {label} thumbnails", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path, f"({Path(save_path).stat().st_size / 1e6:.1f} MB)")


def main():
    device = get_device()
    model, processor, _, _ = load_trained(device)
    model.eval()
    cam = make_cam(model)
    for group in ("65", "34"):
        df = select_examples(group, model, processor, cam)
        print(group, "selected scores:", [f"{s:.2f}" for s in df["score"]])
        render(model, processor, cam, df, group, OUT_DIR / f"gradcam_appendix_{group}_12.png")
        df.to_csv(OUT_DIR / f"samples_appendix_{group}_12.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
