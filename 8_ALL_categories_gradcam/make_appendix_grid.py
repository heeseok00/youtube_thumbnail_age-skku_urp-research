"""부록용 Grad-CAM 격자 생성 (그룹당 12장 = 카테고리당 3장).

50장 격자는 지면에 넣기에 너무 크므로, 본문 Figure 1과 같은 표본에서
카테고리당 상위 3장을 뽑아 원본/예측 클래스 오버레이 쌍으로 배치한다.

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
PER_CATEGORY = 3


def select_examples(group: str) -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / f"samples_correct_{group}_top50.csv")
    picked = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        if len(sub) < PER_CATEGORY:
            raise ValueError(f"{group}/{cat}: {len(sub)} available, need {PER_CATEGORY}")
        picked.append(sub.head(PER_CATEGORY))
    return pd.concat(picked, ignore_index=True)


def render(model, processor, cam, df, group, save_path):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_class = 1 if group == "65" else 0
    ncols = PER_CATEGORY * 2
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), ncols, figsize=(2.6 * ncols, 3.0 * len(CATEGORY_ORDER))
    )
    model.eval()

    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, PER_CATEGORY)
        img_orig = Image.open(row["resolved_path"]).convert("RGB")
        pixel_values = processor(images=img_orig, return_tensors="pt")["pixel_values"].to(
            model_device(model)
        )
        _, _, H, W = pixel_values.shape
        img_resized = img_orig.resize((W, H))
        rgb = np.array(img_resized).astype(np.float32) / 255.0
        heat = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(target_class)])[0]
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


def model_device(model):
    return next(model.parameters()).device


def main():
    device = get_device()
    model, processor, _, _ = load_trained(device)
    cam = make_cam(model)
    for group in ("65", "34"):
        df = select_examples(group)
        render(model, processor, cam, df, group, OUT_DIR / f"gradcam_appendix_{group}_12.png")
        df.to_csv(OUT_DIR / f"samples_appendix_{group}_12.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
