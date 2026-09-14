"""Replace Figure 1 column titles on already-rendered PNGs (no model needed).

Usage:
  python retitle_fig1.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_paper_figures import FIG1_COL_TITLES, OUT_DIR

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
TITLE_BAND = 64
# Image-column centers measured on the current 1891-wide fig1 PNGs.
COL_CENTERS = (360, 896, 1434)
FONT_PX = 40  # ~14.5pt at 200 dpi; long labels still fit the 530px columns


def _font() -> ImageFont.FreeTypeFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold")
        )
        return ImageFont.truetype(path, FONT_PX)
    except Exception:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_PX
        )


def retitle(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGBA")
    w, _ = im.size
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, w, TITLE_BAND), fill=(255, 255, 255, 255))
    font = _font()
    for title, cx in zip(FIG1_COL_TITLES, COL_CENTERS):
        bbox = draw.textbbox((0, 0), title, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, (TITLE_BAND - th) / 2 - 2), title, fill="black", font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    print("Saved:", dst)


def main() -> None:
    pairs = (
        (OUT_DIR / "gradcam_fig1_34.png", PAPER_DIR / "fig1_younger_gradcam.png"),
        (OUT_DIR / "gradcam_fig1_65.png", PAPER_DIR / "fig1_older_gradcam.png"),
    )
    for src, paper_dst in pairs:
        retitle(src, src)
        retitle(src, paper_dst)


if __name__ == "__main__":
    main()
