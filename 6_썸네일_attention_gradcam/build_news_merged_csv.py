"""시사/뉴스/사건 서브카테고리 + DINOv2 768차원 merge CSV 생성."""
from pathlib import Path

import pandas as pd

ROOT = Path("/home/urp_jwl2/26-1_URP")
BIN_ROOT = Path("/home/urp_jwl2/urp_bin/SOCIETY 파일들 - 썸네일 사진 분석")
TARGET_SUBCAT = "시사/뉴스/사건"

INPUT_CSV = ROOT / "Data/SOCIETY/SOCIETY_new_category_v3.csv"
DINOV2_CSV = BIN_ROOT / "SOCIETY_final_kr_clean_with_dinov2_768_fixed.csv"
OUT_CSV = Path(__file__).resolve().parent / "data/SOCIETY_news_with_dinov2.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["subcategory"] == TARGET_SUBCAT].copy()
    print(f"Subcategory '{TARGET_SUBCAT}': {len(df):,} rows")

    dino = pd.read_csv(DINOV2_CSV)
    dino_cols = [c for c in dino.columns if c.startswith("dinov2_")]
    merge_cols = list(dict.fromkeys(["video_id"] + dino_cols))

    df = df.merge(dino[merge_cols], on="video_id", how="inner")
    df = df.loc[:, ~df.columns.duplicated()]
    if "dinov2_image_loaded" in df.columns:
        df = df[df["dinov2_image_loaded"] == True].copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
