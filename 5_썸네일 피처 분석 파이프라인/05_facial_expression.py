"""
05_facial_expression.py
썸네일 인물의 '표정(감정)'을 추출합니다.

의존성 설치:
    pip install fer opencv-python-headless

얼굴이 없는 썸네일(person_count == 0 또는 head_pose == "없음")은
자동으로 건너뛰고 None 처리합니다.

출력 컬럼:
    expression_dominant  - 주요 감정 (분노/혐오/공포/기쁨/슬픔/놀람/중립)
    expression_happy     - 기쁨 확률 (0~1)
    expression_angry     - 분노 확률 (0~1)
    expression_neutral   - 중립 확률 (0~1)
    expression_sad       - 슬픔 확률 (0~1)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_csv, save_csv, resolve_thumbnail_path,
    get_target_rows
)

CHECKPOINT = Path(__file__).parent / "checkpoints/06_facial_expression.csv"
EMOTION_COLS = ["expression_dominant", "expression_happy", "expression_angry",
                "expression_neutral", "expression_sad"]

LABEL_MAP = {
    "angry": "분노", "disgust": "혐오", "fear": "공포",
    "happy": "기쁨", "sad": "슬픔", "surprise": "놀람", "neutral": "중립"
}

NO_FACE_RESULT = {c: None for c in EMOTION_COLS}


def has_face(row: pd.Series) -> bool:
    """앞선 파이프라인 결과로 얼굴 유무를 판단합니다."""
    if "head_pose" in row and pd.notna(row["head_pose"]):
        return row["head_pose"] not in ("없음", "오류")
    if "person_count" in row and pd.notna(row["person_count"]):
        return int(row["person_count"]) > 0
    return True


def extract_expression(img_path: str, detector) -> dict:
    """fer로 얼굴 감정 분석. 가장 큰 얼굴의 감정을 반환."""
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return NO_FACE_RESULT

        results = detector.detect_emotions(img)
        if not results:
            return NO_FACE_RESULT

        # 가장 큰 얼굴(box 넓이 기준) 선택
        largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
        emotions = largest["emotions"]
        dominant = max(emotions, key=emotions.get)

        return {
            "expression_dominant": LABEL_MAP.get(dominant, dominant),
            "expression_happy":    round(emotions.get("happy",   0), 4),
            "expression_angry":    round(emotions.get("angry",   0), 4),
            "expression_neutral":  round(emotions.get("neutral", 0), 4),
            "expression_sad":      round(emotions.get("sad",     0), 4),
        }
    except Exception:
        return NO_FACE_RESULT


def main(args):
    try:
        from fer.fer import FER
    except ImportError:
        print("[설치 필요] pip install fer")
        sys.exit(1)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    target = get_target_rows(df, args.subcategory)
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] '{args.subcategory}' 영상 수: {len(target):,}개")

    no_face_ids = set(
        target.loc[target.apply(lambda r: not has_face(r), axis=1), "video_id"]
    )
    print(f"[사전 필터] 얼굴 없음(스킵): {len(no_face_ids):,}건")

    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(str(CHECKPOINT))
        for _, row in ckpt_df.iterrows():
            val = {c: row.get(c) for c in EMOTION_COLS}
            # 이전에 모두 NaN으로 저장된 잘못된 체크포인트는 무시
            if any(pd.notna(v) for v in val.values()) or row["video_id"] in no_face_ids:
                cached[row["video_id"]] = val
        print(f"[체크포인트] 유효 {len(cached):,}건 불러옴")

    for vid in no_face_ids:
        if vid not in cached:
            cached[vid] = NO_FACE_RESULT

    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    detector = FER(mtcnn=False)
    for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="표정 추출")):
        img_path = resolve_thumbnail_path(row["thumbnail_path"])
        features = extract_expression(str(img_path), detector)
        cached[row["video_id"]] = features

        if (i + 1) % 100 == 0:
            pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    for col in EMOTION_COLS:
        df[col] = df["video_id"].map({k: v[col] for k, v in cached.items()})

    output_path = args.output or args.input.replace(".csv", "_expression.csv")
    save_csv(df, output_path)
    valid = df["expression_dominant"].notna().sum()
    print(f"[결과] 감정 추출 성공: {valid:,}건 / 얼굴 없음: {len(no_face_ids):,}건")
    print(f"[결과] expression_dominant 분포:\n{df['expression_dominant'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="썸네일 인물 표정 추출")
    parser.add_argument("--input", default=str(
        Path(__file__).parent.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
