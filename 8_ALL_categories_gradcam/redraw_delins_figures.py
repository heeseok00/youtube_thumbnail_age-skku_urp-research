"""Redraw deletion/insertion curves with paper-readable type.

Recomputes the same top-50 correct curves as stage_roi, then writes
drop-in Overleaf PNGs (bold 11-12pt labels, no banner titles).

Usage:
  conda run -n urp python redraw_delins_figures.py
  conda run -n urp python redraw_delins_figures.py --from-npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from run_gradcam_pipeline import (
    ANALYSIS_N,
    OUT_DIR,
    get_device,
    load_trained,
    make_cam,
)

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
NPZ_PATH = OUT_DIR / "del_ins_curves.npz"
N_STEPS = 10

STYLE = {
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "axes.linewidth": 1.2,
}


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def compute_group_curves(sample_df, model, processor, cam, device):
    sample_df = sample_df.head(ANALYSIS_N).reset_index(drop=True)
    all_del, all_ins, thrs = [], [], None
    model.eval()
    for i, row in sample_df.iterrows():
        try:
            img_orig = Image.open(row["resolved_path"]).convert("RGB")
            inputs = processor(images=img_orig, return_tensors="pt")
            pv = inputs["pixel_values"].to(device)
            _, _, H, W = pv.shape
            img_resized = img_orig.resize((W, H))
            rgb_np = np.array(img_resized).astype(np.float32) / 255.0
            with torch.no_grad():
                orig_probs = torch.softmax(model(pv), dim=1)[0].cpu().numpy()
            pred_class = int(np.argmax(orig_probs))
            heat = cam(input_tensor=pv, targets=[ClassifierOutputTarget(pred_class)])[0]
            thrs = np.linspace(0, 1, N_STEPS + 1)[1:]
            mean_val = rgb_np.mean(axis=(0, 1), keepdims=True)
            blurred = (
                np.array(img_resized.filter(ImageFilter.GaussianBlur(radius=11))).astype(np.float32)
                / 255.0
            )
            dels, ins = [], []
            for thr in thrs:
                mask = (heat >= np.quantile(heat, 1 - thr)).astype(np.float32)
                del_img = rgb_np * (1 - mask[..., None]) + mean_val * mask[..., None]
                del_pv = processor(
                    images=Image.fromarray((del_img * 255).astype(np.uint8)),
                    return_tensors="pt",
                )["pixel_values"].to(device)
                with torch.no_grad():
                    dels.append(float(torch.softmax(model(del_pv), dim=1)[0, pred_class].cpu()))
                ins_img = rgb_np * mask[..., None] + blurred * (1 - mask[..., None])
                ins_pv = processor(
                    images=Image.fromarray((ins_img * 255).astype(np.uint8)),
                    return_tensors="pt",
                )["pixel_values"].to(device)
                with torch.no_grad():
                    ins.append(float(torch.softmax(model(ins_pv), dim=1)[0, pred_class].cpu()))
            all_del.append(dels)
            all_ins.append(ins)
        except Exception as e:
            print(f"  WARNING skip {i}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample_df)}")
    if not all_del:
        raise RuntimeError("No valid deletion/insertion curves")
    all_del = np.array(all_del)
    all_ins = np.array(all_ins)
    return thrs, all_del, all_ins


def _style_axis(ax, title, xlabel, ylim=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prediction probability")
    ax.grid(alpha=0.3)
    ax.tick_params(width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    if ylim is not None:
        ax.set_ylim(*ylim)


def draw_panel(ax, thrs, curves, color_thin, color_mean, title, xlabel, ylim=None):
    mean_c = curves.mean(0)
    auc = _trapz(mean_c, thrs)
    for curve in curves:
        ax.plot(thrs, curve, color=color_thin, alpha=0.28, linewidth=0.9, zorder=1)
    ax.plot(
        thrs,
        mean_c,
        color=color_mean,
        linewidth=2.8,
        label=f"Mean (AUC = {auc:.3f})",
        zorder=2,
    )
    show = [t for t in thrs if int(round(t * 100)) % 20 == 0]
    ax.set_xticks(show)
    ax.set_xticklabels([f"{int(t * 100)}%" for t in show], rotation=0)
    _style_axis(ax, title, xlabel, ylim=ylim)
    ax.legend(loc="best", handlelength=1.6)
    return auc


def save_pair(path, thrs, all_del, all_ins, group_label):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    auc_d = draw_panel(
        axes[0], thrs, all_del, "salmon", "red",
        f"Deletion, {group_label}", "Fraction deleted", ylim=(0, 1.02),
    )
    auc_i = draw_panel(
        axes[1], thrs, all_ins, "skyblue", "blue",
        f"Insertion, {group_label}", "Fraction revealed",
        ylim=(0, 1.02) if group_label.startswith("65") else None,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path} | DelAUC={auc_d:.3f} InsAUC={auc_i:.3f}")


def save_column(path, thrs, data):
    fig, axes = plt.subplots(4, 1, figsize=(4.4, 12.4))
    specs = [
        (data["del_34"], "salmon", "red", "Deletion, 18–34", "Fraction deleted", (0, 1.02)),
        (data["ins_34"], "skyblue", "blue", "Insertion, 18–34", "Fraction revealed", None),
        (data["del_65"], "salmon", "red", "Deletion, 65+", "Fraction deleted", (0, 1.02)),
        (data["ins_65"], "skyblue", "blue", "Insertion, 65+", "Fraction revealed", (0, 1.02)),
    ]
    for ax, (curves, c1, c2, title, xlabel, ylim) in zip(axes, specs):
        draw_panel(ax, thrs, curves, c1, c2, title, xlabel, ylim=ylim)
    fig.tight_layout(pad=0.6, h_pad=1.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def save_grid(path, thrs, data):
    # Short 2x2 so Table B2 and Figure B1 can share one appendix page.
    # figsize matches AAAI text width; 9–10pt type stays ≥9pt at width=\textwidth.
    with plt.rc_context(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.52), layout="constrained")
        specs = [
            (axes[0, 0], data["del_34"], "salmon", "red", "Deletion, 18–34", "Fraction deleted", (0, 1.02)),
            (axes[0, 1], data["ins_34"], "skyblue", "blue", "Insertion, 18–34", "Fraction revealed", None),
            (axes[1, 0], data["del_65"], "salmon", "red", "Deletion, 65+", "Fraction deleted", (0, 1.02)),
            (axes[1, 1], data["ins_65"], "skyblue", "blue", "Insertion, 65+", "Fraction revealed", (0, 1.02)),
        ]
        for ax, curves, c1, c2, title, xlabel, ylim in specs:
            draw_panel(ax, thrs, curves, c1, c2, title, xlabel, ylim=ylim)
            ax.set_ylabel("Prediction\nprobability")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    print("Saved", path)


def copy_to_paper(src: Path):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPER_DIR / src.name
    dest.write_bytes(src.read_bytes())
    print("Copied", dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-npz", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update(STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_npz and NPZ_PATH.exists():
        z = np.load(NPZ_PATH)
        thrs = z["thrs"]
        data = {
            "del_65": z["del_65"],
            "ins_65": z["ins_65"],
            "del_34": z["del_34"],
            "ins_34": z["ins_34"],
        }
        print("Loaded", NPZ_PATH)
    else:
        device = get_device()
        print("device:", device)
        model, processor, _, _ = load_trained(device)
        cam = make_cam(model)
        correct_65 = pd.read_csv(OUT_DIR / "samples_correct_65_top50.csv")
        correct_34 = pd.read_csv(OUT_DIR / "samples_correct_34_top50.csv")
        print("[65+] deletion/insertion")
        thrs, del_65, ins_65 = compute_group_curves(correct_65, model, processor, cam, device)
        print("[18-34] deletion/insertion")
        thrs, del_34, ins_34 = compute_group_curves(correct_34, model, processor, cam, device)
        np.savez(
            NPZ_PATH,
            thrs=thrs,
            del_65=del_65,
            ins_65=ins_65,
            del_34=del_34,
            ins_34=ins_34,
        )
        print("Saved", NPZ_PATH)
        data = {
            "del_65": del_65,
            "ins_65": ins_65,
            "del_34": del_34,
            "ins_34": ins_34,
        }

    older = OUT_DIR / "figA1_older_delins.png"
    younger = OUT_DIR / "figA1_younger_delins.png"
    column = OUT_DIR / "figB1_delins_column.png"
    grid = OUT_DIR / "figB1_delins_2x2.png"
    save_pair(older, thrs, data["del_65"], data["ins_65"], "65+")
    save_pair(younger, thrs, data["del_34"], data["ins_34"], "18–34")
    save_column(column, thrs, data)
    save_grid(grid, thrs, data)
    for p in (older, younger, column, grid):
        copy_to_paper(p)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
