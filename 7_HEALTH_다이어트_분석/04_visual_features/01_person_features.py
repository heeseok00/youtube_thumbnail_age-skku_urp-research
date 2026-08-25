"""
01_person_features.py
다이어트 샘플 썸네일에서 '인물 비중'과 '인물 수'를 YOLOv8s-seg 한 번 추론으로 추출.
(원본: 5_썸네일 피처 분석 파이프라인/01_person_features.py — 데이터 로딩만 변경)

출력 컬럼: person_ratio, person_count, person_count_cat
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, load_ckpt, save_ckpt, CKPT_DIR

CHECKPOINT = CKPT_DIR / "01_person_features.csv"
WEIGHTS = Path(__file__).resolve().parent.parent.parent / "yolov8s-seg.pt"
COLS = ["person_ratio", "person_count"]


def extract_person_features(img_path: str, model) -> dict:
    """인물 마스크 합집합 면적(원본 해상도 기준) / 전체 픽셀 + 인물 수."""
    try:
        import torch.nn.functional as F

        results = model(img_path, classes=[0], verbose=False, conf=0.2)
        result = results[0]
        h, w = result.orig_shape
        total_pixels = h * w

        if result.masks is None or len(result.masks) == 0:
            return {"person_ratio": 0.0, "person_count": 0}

        masks = result.masks.data.float()  # (N, mh, mw) — 추론 해상도
        count = int(masks.shape[0])
        masks_orig = F.interpolate(
            masks.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        )[0]
        union = (masks_orig > 0.5).any(dim=0)
        ratio = float(union.sum().item() / total_pixels) if total_pixels > 0 else 0.0
        return {"person_ratio": round(min(ratio, 1.0), 6), "person_count": count}

    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return {"person_ratio": float("nan"), "person_count": -1}


def categorize_count(n) -> str | None:
    if pd.isna(n) or n < 0:
        return None
    n = int(n)
    if n == 0:   return "0명"
    elif n == 1: return "1명"
    elif n == 2: return "2명"
    else:        return "3명+"


def main(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[설치 필요] pip install ultralytics")
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
        weights = str(WEIGHTS) if WEIGHTS.exists() else "yolov8s-seg.pt"
        model = YOLO(weights)
        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="인물 피처 추출")):
            cached[row["video_id"]] = extract_person_features(row["resolved_path"], model)
            if (i + 1) % 100 == 0:
                save_ckpt(cached, CHECKPOINT)
        save_ckpt(cached, CHECKPOINT)

    done = pd.Series({k: v["person_count"] for k, v in cached.items()})
    print(f"[결과] 처리 완료: {len(cached):,}건")
    print(f"[결과] person_count 분포:\n{done.value_counts().sort_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 인물 비중 + 인물 수 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
