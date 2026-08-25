"""
06_merge_features.py
01~05 체크포인트를 다이어트 샘플에 병합해 최종 피처 CSV를 생성.

출력: outputs/diet_visual_features.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, CKPT_DIR, OUT_DIR

CKPTS = [
    ("01_person_features.csv",   ["person_ratio", "person_count"]),
    ("02_text_ratio.csv",        ["text_ratio"]),
    ("03_color_distribution.csv", ["color_hue_mean", "color_saturation", "color_brightness",
                                   "color_warm_ratio", "color_entropy", "color_hue_std"]),
    ("03b_text_region_color.csv", ["text_color_entropy", "text_color_saturation",
                                   "text_color_brightness", "text_color_hue_std",
                                   "bg_color_entropy", "text_bg_entropy_diff",
                                   "text_bg_saturation_diff"]),
    ("04_head_pose.csv",         ["head_pose", "head_pose_face_count"]),
    ("05_facial_expression.csv", ["expression_dominant", "expression_happy", "expression_angry",
                                  "expression_neutral", "expression_sad"]),
]


def categorize_count(n):
    if pd.isna(n) or n < 0:
        return None
    n = int(n)
    if n == 0:   return "0명"
    elif n == 1: return "1명"
    elif n == 2: return "2명"
    else:        return "3명+"


def main():
    df = load_sample()
    print(f"[샘플] {len(df):,}행")

    for fname, cols in CKPTS:
        path = CKPT_DIR / fname
        if not path.exists():
            print(f"[스킵] {fname} 없음")
            continue
        ckpt = pd.read_csv(path)
        avail = [c for c in cols if c in ckpt.columns]
        df = df.merge(ckpt[["video_id"] + avail], on="video_id", how="left")
        done = ckpt[avail[0]].notna().sum() if avail else 0
        print(f"[병합] {fname}: {len(ckpt):,}행 (첫 컬럼 non-null {done:,})")

    if "person_count" in df.columns:
        df["person_count_cat"] = df["person_count"].apply(categorize_count)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "diet_visual_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path} ({len(df):,}행, {len(df.columns)}컬럼)")

    feat_cols = [c for name, cols in CKPTS for c in cols if c in df.columns]
    print("\n피처 non-null 현황:")
    for c in feat_cols:
        print(f"  {c}: {df[c].notna().sum():,}")


if __name__ == "__main__":
    main()
