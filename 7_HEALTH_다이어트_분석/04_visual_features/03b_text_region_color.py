"""
03b_text_region_color.py
다이어트 샘플 썸네일에서 EasyOCR 텍스트 ROI / 배경 ROI 색상 피처를 추출.
(원본: 5_썸네일 피처 분석 파이프라인/03b_text_region_color.py — 데이터 로딩만 변경)

출력 컬럼:
    text_color_entropy, text_color_saturation, text_color_brightness,
    text_color_hue_std, bg_color_entropy,
    text_bg_entropy_diff, text_bg_saturation_diff
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import circstd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, load_ckpt, save_ckpt, CKPT_DIR

OCR_TIMEOUT = 30
CHECKPOINT = CKPT_DIR / "03b_text_region_color.csv"
COLS = [
    "text_color_entropy",
    "text_color_saturation",
    "text_color_brightness",
    "text_color_hue_std",
    "bg_color_entropy",
    "text_bg_entropy_diff",
    "text_bg_saturation_diff",
]
NAN_RESULT = {c: float("nan") for c in COLS}


def rgb_to_hsv_arrays(img_array: np.ndarray):
    """RGB float32 (0~255) → hue_cv(0~180), sat(0~1), val(0~1)."""
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
    hue_cv = hue / 2

    sat = np.zeros_like(r)
    sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    val = cmax
    return hue_cv, sat, val


def hue_entropy(hue_cv, sat, region_mask, min_colored: int = 10) -> float:
    colored = hue_cv[region_mask & (sat > 0.15)]
    if len(colored) <= min_colored:
        return 0.0
    hist, _ = np.histogram(colored, bins=18, range=(0, 180))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def region_color_features(hue_cv, sat, val, region_mask: np.ndarray) -> dict:
    if region_mask.sum() < 10:
        return {k: float("nan") for k in ("entropy", "saturation", "brightness", "hue_std")}

    sat_roi = sat[region_mask]
    val_roi = val[region_mask]
    hue_roi = hue_cv[region_mask]
    hue_rad = np.deg2rad(hue_roi * 2)
    hue_std = float(np.rad2deg(circstd(hue_rad, high=2 * np.pi, low=0)) / 2) if len(hue_rad) > 1 else 0.0

    return {
        "entropy": hue_entropy(hue_cv, sat, region_mask),
        "saturation": round(float(np.mean(sat_roi)) * 255, 2),
        "brightness": round(float(np.mean(val_roi)) * 255, 2),
        "hue_std": round(hue_std, 2),
    }


def build_text_mask(img_array: np.ndarray, ocr_results) -> np.ndarray:
    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box, _, _ in ocr_results:
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    if mask.any():
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def extract_text_region_color_features(img_path: str, ocr_reader) -> dict:
    try:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        results = ocr_reader.readtext(img_path)

        hue_cv, sat, val = rgb_to_hsv_arrays(img_array)

        if not results:
            bg = region_color_features(hue_cv, sat, val, np.ones_like(sat, dtype=bool))
            return {
                "text_color_entropy": float("nan"),
                "text_color_saturation": float("nan"),
                "text_color_brightness": float("nan"),
                "text_color_hue_std": float("nan"),
                "bg_color_entropy": round(bg["entropy"], 4),
                "text_bg_entropy_diff": float("nan"),
                "text_bg_saturation_diff": float("nan"),
            }

        text_mask = build_text_mask(img_array, results)
        bg_mask = ~text_mask

        text = region_color_features(hue_cv, sat, val, text_mask)
        bg = region_color_features(hue_cv, sat, val, bg_mask)

        return {
            "text_color_entropy": round(text["entropy"], 4),
            "text_color_saturation": text["saturation"],
            "text_color_brightness": text["brightness"],
            "text_color_hue_std": text["hue_std"],
            "bg_color_entropy": round(bg["entropy"], 4),
            "text_bg_entropy_diff": round(text["entropy"] - bg["entropy"], 4),
            "text_bg_saturation_diff": round(text["saturation"] - bg["saturation"], 2),
        }
    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return NAN_RESULT.copy()


def _persistent_worker(in_q, out_q):
    import easyocr

    ocr = easyocr.Reader(["ko", "en"], gpu=True, verbose=False)
    while True:
        img_path = in_q.get()
        if img_path is None:
            break
        out_q.put(extract_text_region_color_features(img_path, ocr))


class PersistentTextColorOCR:
    def __init__(self):
        self._start_worker()

    def _start_worker(self):
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.worker = mp.Process(target=_persistent_worker, args=(self.in_q, self.out_q), daemon=True)
        self.worker.start()

    def extract(self, img_path: str) -> dict:
        self.in_q.put(img_path)
        try:
            return self.out_q.get(timeout=OCR_TIMEOUT)
        except Exception:
            print(f"\n[타임아웃] {img_path} — 워커 재시작")
            self.worker.terminate()
            self.worker.join()
            self._start_worker()
            return NAN_RESULT.copy()

    def close(self):
        try:
            self.in_q.put(None)
            self.worker.join(timeout=5)
        except Exception:
            pass


def main(args):
    try:
        import easyocr  # noqa: F401
    except ImportError:
        print("[설치 필요] pip install easyocr opencv-python-headless")
        sys.exit(1)

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

    if len(remaining) > 0:
        worker = PersistentTextColorOCR()
        try:
            for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="텍스트 영역 색상")):
                cached[row["video_id"]] = worker.extract(row["resolved_path"])
                if (i + 1) % 50 == 0:
                    save_ckpt(cached, CHECKPOINT)
        finally:
            worker.close()
        save_ckpt(cached, CHECKPOINT)

    print(f"[결과] 처리 완료: {len(cached):,}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 텍스트/배경 ROI 색상 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
