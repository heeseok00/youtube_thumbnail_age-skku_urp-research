"""논문용 Grad-CAM 그림 생성.

본문 Figure 1 : 카테고리당 1장 (원본 / 오버레이 / 히트맵)
부록 Figure A2, A3 : 카테고리당 3장 (원본 / 오버레이)

두 그림 모두 채널 중복을 제거한 정분류 상위 50장에서, 각 집단의 전형적인
양상이 가장 뚜렷한 사례를 고른다.
  65+ : 히트맵 에너지의 text + person 비율이 가장 높은 이미지
  ~34 : 히트맵이 가장 넓게 퍼진 이미지 (에너지 50%를 담는 데 필요한 면적 비율)

Usage:
  python make_paper_figures.py
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
GROUP_LABEL = {"65": "65+", "34": "~34"}
LABEL_KW = {"fontsize": 15, "fontweight": "bold"}
ROW_KW = {"fontsize": 17, "fontweight": "bold"}


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


def score_candidates(group, model, processor, cam):
    """상위 50장에 전형성 점수를 매기고 카테고리별로 정렬한다."""
    df = pd.read_csv(OUT_DIR / "cam_roi_share_per_image.csv")
    df = df[df["group"] == GROUP_KEY[group]].copy()
    if group == "65":
        df["score"] = df["share_text"].fillna(0) + df["share_person"].fillna(0)
    else:
        df["score"] = [
            energy_half_area(compute_cam(model, processor, cam, p, 0)[1])
            for p in df["resolved_path"]
        ]
    df["category"] = pd.Categorical(df["category"], CATEGORY_ORDER, ordered=True)
    return df.sort_values(["category", "score"], ascending=[True, False])


def top_per_category(scored, n):
    picked = scored.groupby("category", observed=True).head(n)
    missing = {c for c in CATEGORY_ORDER if (picked["category"] == c).sum() < n}
    if missing:
        raise ValueError(f"Not enough candidates for {sorted(missing)} (need {n})")
    return picked.sort_values(["category", "score"], ascending=[True, False]).reset_index(drop=True)


def strip_axes(*axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def render_main(model, processor, cam, df, group, save_path):
    """본문 Figure 1: 카테고리당 1장, 원본 / 오버레이 / 히트맵."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_class = 1 if group == "65" else 0
    fig, axes = plt.subplots(len(CATEGORY_ORDER), 3, figsize=(9.6, 3.1 * len(CATEGORY_ORDER)))

    for r, (_, row) in enumerate(df.iterrows()):
        img, heat = compute_cam(model, processor, cam, row["resolved_path"], target_class)
        rgb = np.array(img).astype(np.float32) / 255.0
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        im = axes[r, 2].imshow(heat, cmap="jet", vmin=0, vmax=1)
        strip_axes(*axes[r])
        axes[r, 0].set_ylabel(CATEGORY_LABEL[row["category"]], labelpad=10, **ROW_KW)
        if r == 0:
            axes[r, 0].set_title("original", **LABEL_KW)
            axes[r, 1].set_title("Grad-CAM overlay", **LABEL_KW)
            axes[r, 2].set_title("heatmap", **LABEL_KW)

    # 컬러바를 별도 축에 그린다. ax=axes[:, 2]로 붙이면 히트맵 열만 좁아진다.
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cbar = fig.colorbar(im, cax=fig.add_axes((0.92, 0.3, 0.018, 0.4)))
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path, f"({Path(save_path).stat().st_size / 1e6:.1f} MB)")


def render_appendix(model, processor, cam, df, group, save_path, per_category=3):
    """부록 Figure A2/A3: 카테고리당 여러 장, 원본 / 오버레이 쌍."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_class = 1 if group == "65" else 0
    ncols = per_category * 2
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), ncols, figsize=(2.6 * ncols, 3.0 * len(CATEGORY_ORDER))
    )

    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, per_category)
        img, heat = compute_cam(model, processor, cam, row["resolved_path"], target_class)
        rgb = np.array(img).astype(np.float32) / 255.0
        ax_o, ax_c = axes[r, c * 2], axes[r, c * 2 + 1]
        ax_o.imshow(img)
        ax_c.imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        strip_axes(ax_o, ax_c)
        if c == 0:
            ax_o.set_ylabel(CATEGORY_LABEL[row["category"]], labelpad=10, **ROW_KW)
        ax_o.set_title("original", **LABEL_KW)
        ax_c.set_title("Grad-CAM", **LABEL_KW)

    fig.suptitle(
        f"Grad-CAM for correctly classified {GROUP_LABEL[group]} thumbnails",
        fontsize=20,
        fontweight="bold",
    )
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
        scored = score_candidates(group, model, processor, cam)

        main_df = top_per_category(scored, 1)
        print(group, "figure 1 scores:", [f"{s:.2f}" for s in main_df["score"]])
        render_main(model, processor, cam, main_df, group, OUT_DIR / f"gradcam_fig1_{group}.png")
        main_df.to_csv(
            OUT_DIR / f"samples_fig1_{group}.csv", index=False, encoding="utf-8-sig"
        )

        app_df = top_per_category(scored, 3)
        print(group, "appendix scores:", [f"{s:.2f}" for s in app_df["score"]])
        render_appendix(
            model, processor, cam, app_df, group, OUT_DIR / f"gradcam_appendix_{group}_12.png"
        )
        app_df.to_csv(
            OUT_DIR / f"samples_appendix_{group}_12.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()
