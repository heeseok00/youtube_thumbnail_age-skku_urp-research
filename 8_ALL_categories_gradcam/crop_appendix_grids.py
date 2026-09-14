"""Drop the first original/Grad-CAM pair from appendix grids (duplicates Figure 1).

Usage:
  python crop_appendix_grids.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_paper_figures import GROUP_LABEL, OUT_DIR

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
# Keep the EDU/HEALTH/... strip, then pairs 2 and 3 (measured on the 3099-wide PNGs).
YLABEL_END = 90
PAIR2_START = 1075
TITLE_BAND = 70
TITLE_PX = 52


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold")
        )
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )


def crop_grid(src: Path, dst: Path, group: str) -> None:
    im = Image.open(src).convert("RGBA")
    left = im.crop((0, 0, YLABEL_END, im.height))
    right = im.crop((PAIR2_START, 0, im.width, im.height))
    out = Image.new("RGBA", (left.width + right.width, im.height), (255, 255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, TITLE_BAND), fill=(255, 255, 255, 255))
    title = f"Grad-CAM for correctly classified {GROUP_LABEL[group]} thumbnails"
    font = _font(TITLE_PX)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((out.width - tw) / 2, (TITLE_BAND - th) / 2), title, fill="black", font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    print("Saved:", dst, out.size)


def main() -> None:
    jobs = (
        ("34", OUT_DIR / "gradcam_appendix_34_12.png", PAPER_DIR / "figA3_younger_grid.png"),
        ("65", OUT_DIR / "gradcam_appendix_65_12.png", PAPER_DIR / "figA2_older_grid.png"),
    )
    for group, src, paper_dst in jobs:
        crop_grid(src, paper_dst, group)
        crop_grid(src, OUT_DIR / f"gradcam_appendix_{group}_8.png", group)


if __name__ == "__main__":
    main()
