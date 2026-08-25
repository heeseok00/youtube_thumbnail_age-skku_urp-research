"""
05_facial_expression.py
다이어트 샘플 썸네일 인물의 '표정(감정)'을 추출 (FER).
(원본: 5_썸네일 피처 분석 파이프라인/05_facial_expression.py — 데이터 로딩만 변경)

04_gaze_direction의 체크포인트가 있으면 head_pose == "없음"인 썸네일은
FER 추론 없이 자동 스킵합니다.

출력 컬럼:
    expression_dominant, expression_happy, expression_angry,
    expression_neutral, expression_sad
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, save_ckpt, CKPT_DIR

CHECKPOINT = CKPT_DIR / "05_facial_expression.csv"
HEAD_POSE_CKPT = CKPT_DIR / "04_head_pose.csv"
EMOTION_COLS = ["expression_dominant", "expression_happy", "expression_angry",
                "expression_neutral", "expression_sad"]

LABEL_MAP = {
    "angry": "분노", "disgust": "혐오", "fear": "공포",
    "happy": "기쁨", "sad": "슬픔", "surprise": "놀람", "neutral": "중립"
}

NO_FACE_RESULT = {c: None for c in EMOTION_COLS}


def get_no_face_ids() -> set:
    """04 단계 결과에서 얼굴 없는 video_id 집합을 가져옴 (없으면 빈 집합)."""
    if not HEAD_POSE_CKPT.exists():
        print("[안내] head_pose 체크포인트 없음 — 전체 이미지에 FER 실행")
        return set()
    hp = pd.read_csv(HEAD_POSE_CKPT)
    return set(hp.loc[hp["head_pose"] == "없음", "video_id"])


def extract_expression(img_path: str, detector) -> dict:
    """가장 큰 얼굴의 감정을 반환."""
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return NO_FACE_RESULT

        results = detector.detect_emotions(img)
        if not results:
            return NO_FACE_RESULT

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
        print("[설치 필요] pip install fer opencv-python-headless")
        sys.exit(1)

    target = load_sample()
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] 다이어트 샘플: {len(target):,}개")

    no_face_ids = get_no_face_ids() & set(target["video_id"])
    print(f"[사전 필터] 얼굴 없음(스킵): {len(no_face_ids):,}건")

    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(CHECKPOINT)
        for _, row in ckpt_df.iterrows():
            val = {c: row.get(c) for c in EMOTION_COLS}
            # 전부 NaN으로 저장된 잘못된 행은 얼굴 없음으로 확인된 경우만 유지
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
        cached[row["video_id"]] = extract_expression(row["resolved_path"], detector)
        if (i + 1) % 100 == 0:
            save_ckpt(cached, CHECKPOINT)
    save_ckpt(cached, CHECKPOINT)

    dom = pd.Series({k: v["expression_dominant"] for k, v in cached.items()})
    print(f"[결과] 감정 추출 성공: {dom.notna().sum():,}건 / 얼굴 없음: {len(no_face_ids):,}건")
    print(f"[결과] expression_dominant 분포:\n{dom.value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 인물 표정 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
