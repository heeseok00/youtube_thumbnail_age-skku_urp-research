"""
03_color_distribution.py
다이어트 샘플 썸네일의 '색상 분포'를 추출 (PIL + numpy, GPU 불필요).
(원본: 5_썸네일 피처 분석 파이프라인/03_color_distribution.py — 데이터 로딩만 변경)

출력 컬럼:
    color_hue_mean, color_saturation, color_brightness,
    color_warm_ratio, color_entropy, color_hue_std
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import circmean, circstd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, load_ckpt, save_ckpt, CKPT_DIR

CHECKPOINT = CKPT_DIR / "03_color_distribution.csv"
COLS = ["color_hue_mean", "color_saturation", "color_brightness",
        "color_warm_ratio", "color_entropy", "color_hue_std"]


def extract_color_features(img_path: str) -> dict:
    try:
        img = Image.open(img_path).convert("RGB").resize((160, 90))
        img_array = np.array(img, dtype=np.float32)

        r = img_array[:, :, 0] / 255.0
        g = img_array[:, :, 1] / 255.0
        b = img_array[:, :, 2] / 255.0
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        hue = np.zeros_like(r)
        mask = delta != 0
        mr = mask & (cmax == r)
        mg = mask & (cmax == g)
        mb = mask & (cmax == b)
        hue[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
        hue[mg] = 60 * ((b[mg] - r[mg]) / delta[mg]) + 120
        hue[mb] = 60 * ((r[mb] - g[mb]) / delta[mb]) + 240
        hue_cv = hue / 2  # OpenCV 스케일 (0~180)

        sat = np.zeros_like(r)
        sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
        val = cmax

        hue_rad = np.deg2rad(hue_cv * 2)
        hue_mean_cv = float(np.rad2deg(circmean(hue_rad.ravel(), high=2*np.pi, low=0)) / 2)
        hue_std_cv = float(np.rad2deg(circstd(hue_rad.ravel(), high=2*np.pi, low=0)) / 2)

        warm_mask = ((hue_cv >= 0) & (hue_cv <= 30)) | ((hue_cv >= 150) & (hue_cv <= 180))
        warm_ratio = float(warm_mask.sum() / warm_mask.size)

        # 무채색(채도 낮은 픽셀) 제외한 hue 히스토그램 entropy
        colored = hue_cv[sat > 0.15]
        if len(colored) > 10:
            hist, _ = np.histogram(colored, bins=18, range=(0, 180))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = float(-np.sum(hist * np.log2(hist)))
        else:
            entropy = 0.0

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
    target = load_sample()
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] 다이어트 샘플: {len(target):,}개")

    cached = load_ckpt(CHECKPOINT, COLS)
    if cached:
        print(f"[체크포인트] {len(cached):,}건 불러옴")
    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="색상 분포 추출")):
        cached[row["video_id"]] = extract_color_features(row["resolved_path"])
        if (i + 1) % 200 == 0:
            save_ckpt(cached, CHECKPOINT)
    save_ckpt(cached, CHECKPOINT)

    print(f"[결과] 처리 완료: {len(cached):,}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 색상 분포 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
