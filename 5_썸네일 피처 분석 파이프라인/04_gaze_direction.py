"""
05_gaze_direction.py
썸네일 인물의 '머리 방향(head pose)'을 추출합니다.
mediapipe 0.10+ Tasks API 사용

의존성: pip install mediapipe (이미 설치됨)
모델: face_landmarker.task (최초 실행 시 자동 다운로드)

출력 컬럼:
    head_pose             - 정면 / 측면 / 없음
    head_pose_face_count  - 검출된 얼굴 수
"""

import argparse
import sys
import urllib.request
from pathlib import Path
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_csv, save_csv, resolve_thumbnail_path, get_target_rows

CHECKPOINT = Path(__file__).parent / "checkpoints/05_head_pose.csv"
MODEL_PATH  = Path(__file__).parent / "face_landmarker.task"
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
COLS        = ["head_pose", "head_pose_face_count"]
YAW_THRESHOLD = 20


def download_model():
    if not MODEL_PATH.exists():
        print(f"[모델 다운로드] {MODEL_URL}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[완료] face_landmarker.task 다운로드됨")


def estimate_yaw(landmarks, w: int, h: int) -> float:
    nose      = landmarks[1]
    left_ear  = landmarks[234]
    right_ear = landmarks[454]
    nose_x  = nose.x * w
    left_x  = left_ear.x * w
    right_x = right_ear.x * w
    face_w  = right_x - left_x
    if face_w == 0:
        return 0.0
    return float(((nose_x - (left_x + right_x) / 2) / (face_w / 2)) * 90)


def face_bbox_area(landmarks, w: int, h: int) -> float:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def extract_head_pose(img_path: str, detector) -> dict:
    try:
        import mediapipe as mp
        image = mp.Image.create_from_file(img_path)
        result = detector.detect(image)

        if not result.face_landmarks:
            return {"head_pose": "없음", "head_pose_face_count": 0}

        face_count = len(result.face_landmarks)
        w, h = image.width, image.height

        # 가장 큰 얼굴(주 인물) 선택
        largest = max(result.face_landmarks, key=lambda lm: face_bbox_area(lm, w, h))
        yaw = estimate_yaw(largest, w, h)
        direction = "정면" if abs(yaw) <= YAW_THRESHOLD else "측면"

        return {"head_pose": direction, "head_pose_face_count": face_count}
    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return {"head_pose": "오류", "head_pose_face_count": -1}


def main(args):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("[설치 필요] pip install mediapipe")
        sys.exit(1)

    download_model()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    target = get_target_rows(df, args.subcategory)
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] \'{args.subcategory}\' 영상 수: {len(target):,}개")

    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(str(CHECKPOINT))
        for _, row in ckpt_df.iterrows():
            cached[row["video_id"]] = {c: row.get(c) for c in COLS}
        print(f"[체크포인트] {len(cached):,}건 불러옴")

    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    if len(remaining) > 0:
        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            min_face_detection_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)

        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="머리 방향 추출")):
            img_path = resolve_thumbnail_path(row["thumbnail_path"])
            features = extract_head_pose(str(img_path), detector)
            cached[row["video_id"]] = features

            if (i + 1) % 100 == 0:
                pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

        detector.close()
        pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(str(CHECKPOINT), index=False)

    for col in COLS:
        df[col] = df["video_id"].map({k: v[col] for k, v in cached.items()})

    output_path = args.output or args.input.replace(".csv", "_headpose.csv")
    save_csv(df, output_path)
    col = 'head_pose'
    print(f'[결과] head_pose 분포:\n' + str(df[col].value_counts()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(
        Path(__file__).parent.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
