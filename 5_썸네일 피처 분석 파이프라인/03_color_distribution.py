"""
03_color_distribution.py
썸네일의 '색상 분포'를 추출합니다.

추가 설치 불필요 (PIL, numpy, scipy 사용)

출력 컬럼:
    color_hue_mean      - 순환 평균 색조 (H, 0~180)
    color_saturation    - 평균 채도 (0~255)
    color_brightness    - 평균 밝기 (0~255)
    color_warm_ratio    - 따뜻한 색 비율 (H 0~30, 150~180: 빨강/주황/분홍)
    color_entropy       - 색상 다양성 (hue 히스토그램 Shannon entropy, 높을수록 다양)
    color_hue_std       - 색조 분산 (순환 표준편차, 높을수록 다양)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.stats import circmean, circstd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_csv, save_csv, resolve_thumbnail_path,
    get_target_rows
)

CHECKPOINT = Path(__file__).parent / "checkpoints/03_color_distribution.csv"
COLS = ["color_hue_mean", "color_saturation", "color_brightness",
        "color_warm_ratio", "color_entropy", "color_hue_std"]


def extract_color_features(img_path: str) -> dict:
    """PIL + numpy로 HSV 색상 특성 추출."""
    try:
        img = Image.open(img_path).convert("RGB").resize((160, 90))
        img_array = np.array(img, dtype=np.float32)

        r = img_array[:, :, 0] / 255.0
        g = img_array[:, :, 1] / 255.0
        b = img_array[:, :, 2] / 255.0
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue (0~360도)
        hue = np.zeros_like(r)
        mask = delta != 0
        mr = mask & (cmax == r)
        mg = mask & (cmax == g)
        mb = mask & (cmax == b)
        hue[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
        hue[mg] = 60 * ((b[mg] - r[mg]) / delta[mg]) + 120
        hue[mb] = 60 * ((r[mb] - g[mb]) / delta[mb]) + 240
        hue_cv = hue / 2  # OpenCV 스케일 (0~180)

        # Saturation, Brightness
        sat = np.zeros_like(r)
        sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
        val = cmax

        # 순환 평균/표준편차 (circmean/circstd는 라디안 단위 → 도 변환 필요)
        hue_rad = np.deg2rad(hue_cv * 2)  # 0~180 → 0~360 → 라디안
        hue_mean_rad = circmean(hue_rad.ravel(), high=2*np.pi, low=0)
        hue_std_rad  = circstd(hue_rad.ravel(), high=2*np.pi, low=0)
        hue_mean_cv  = float(np.rad2deg(hue_mean_rad) / 2)  # 다시 0~180으로
        hue_std_cv   = float(np.rad2deg(hue_std_rad) / 2)

        # 따뜻한 색 비율: H 0~30 (빨강/주황) + H 150~180 (분홍/자주)
        warm_mask = ((hue_cv >= 0) & (hue_cv <= 30)) | ((hue_cv >= 150) & (hue_cv <= 180))
        warm_ratio = float(warm_mask.sum() / warm_mask.size)

        # 색상 다양성 - hue 히스토그램 Shannon entropy
        # 채도가 낮은 픽셀(무채색)은 제외하여 실제 색상만 반영
        colored = hue_cv[sat > 0.15]  # 채도 15% 이상만
        if len(colored) > 10:
            hist, _ = np.histogram(colored, bins=18, range=(0, 180))  # 10도 단위 18구간
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = float(-np.sum(hist * np.log2(hist)))
        else:
            entropy = 0.0  # 무채색 이미지

        return {
            "color_hue_mean":   round(hue_mean_cv, 2),
            "color_saturation": round(float(np.mean(sat)) * 255, 2),
            "color_brightness": round(float(np.mean(val)) * 255, 2),
            "color_warm_ratio": round(warm_ratio, 4),
            "color_entropy":    round(entropy, 4),
            "color_hue_std":    round(hue_std_cv, 2),
        }
    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return {c: float("nan") for c in COLS}


def main(args):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    target = get_target_rows(df, args.subcategory)
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] '{args.subcategory}' 영상 수: {len(target):,}개")

    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(str(CHECKPOINT))
        for _, row in ckpt_df.iterrows():
            cached[row["video_id"]] = {c: row[c] for c in COLS}
        print(f"[체크포인트] {len(cached):,}건 불러옴")

    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="색상 분포 추출")):
        img_path = resolve_thumbnail_path(row["thumbnail_path"])
        features = extract_color_features(str(img_path))
        cached[row["video_id"]] = features

        if (i + 1) % 200 == 0:
            pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    for col in COLS:
        df[col] = df["video_id"].map({k: v[col] for k, v in cached.items()})

    output_path = args.output or args.input.replace(".csv", "_color.csv")
    save_csv(df, output_path)
    for col in COLS:
        print(f"[결과] {col} 비null: {df[col].notna().sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="썸네일 색상 분포 추출")
    parser.add_argument("--input", default=str(
        Path(__file__).parent.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
