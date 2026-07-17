"""
01_person_features.py
썸네일에서 '인물 비중'과 '인물 수'를 한 번의 YOLOv8 추론으로 동시 추출합니다.
(기존 01_person_ratio.py + 04_person_count.py 통합)

의존성 설치:
    pip install ultralytics

출력 컬럼:
    person_ratio      - 인물 면적 비율 (0.0 ~ 1.0)
    person_count      - 인물 수 (0, 1, 2, ...)
    person_count_cat  - 인물 수 범주 (0명 / 1명 / 2명 / 3명+)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_csv, save_csv, resolve_thumbnail_path,
    get_target_rows
)

CHECKPOINT = Path(__file__).parent / "checkpoints/01_person_features.csv"
COLS = ["person_ratio", "person_count"]


def extract_person_features(img_path: str, model) -> dict:
    """YOLOv8s-seg 한 번 추론으로 인물 비중 + 인물 수 동시 추출."""
    try:
        results = model(img_path, classes=[0], verbose=False, conf=0.2)
        result = results[0]
        h, w = result.orig_shape
        total_pixels = h * w

        count = 0
        person_pixels = 0

        if result.masks is not None and len(result.masks) > 0:
            count = len(result.masks)
            for mask in result.masks.data:
                person_pixels += int(mask.cpu().numpy().sum())

        ratio = min(person_pixels / total_pixels, 1.0) if total_pixels > 0 else 0.0
        return {"person_ratio": round(ratio, 6), "person_count": count}

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

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    target = get_target_rows(df, args.subcategory)
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] '{args.subcategory}' 영상 수: {len(target):,}개")

    # 체크포인트 로드
    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(str(CHECKPOINT))
        for _, row in ckpt_df.iterrows():
            cached[row["video_id"]] = {c: row[c] for c in COLS}
        print(f"[체크포인트] {len(cached):,}건 불러옴")

    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    if len(remaining) > 0:
        model = YOLO("yolov8s-seg.pt")  # nano → small 업그레이드
        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="인물 피처 추출")):
            img_path = resolve_thumbnail_path(row["thumbnail_path"])
            features = extract_person_features(str(img_path), model)
            cached[row["video_id"]] = features

            if (i + 1) % 100 == 0:
                pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

        pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    for col in COLS:
        df[col] = df["video_id"].map({k: v[col] for k, v in cached.items()})
    df["person_count_cat"] = df["person_count"].apply(categorize_count)

    output_path = args.output or args.input.replace(".csv", "_person.csv")
    save_csv(df, output_path)
    print(f"[결과] person_ratio 비null: {df['person_ratio'].notna().sum():,}")
    print(f"[결과] person_count 분포:\n{df['person_count'].value_counts().sort_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="썸네일 인물 비중 + 인물 수 동시 추출")
    parser.add_argument("--input", default=str(
        Path(__file__).parent.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
