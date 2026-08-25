"""
04_gaze_direction.py
다이어트 샘플 썸네일 인물의 '머리 방향(head pose)'을 추출 (MediaPipe Tasks API).
(원본: 5_썸네일 피처 분석 파이프라인/04_gaze_direction.py — 데이터 로딩만 변경)

출력 컬럼: head_pose (정면/측면/없음), head_pose_face_count
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, load_ckpt, save_ckpt, CKPT_DIR

CHECKPOINT = CKPT_DIR / "04_head_pose.csv"
# 기존 파이프라인에서 다운로드해 둔 모델 재사용
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "5_썸네일 피처 분석 파이프라인/face_landmarker.task"
LOCAL_MODEL = Path(__file__).parent / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
COLS = ["head_pose", "head_pose_face_count"]
YAW_THRESHOLD = 20


def get_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if not LOCAL_MODEL.exists():
        print(f"[모델 다운로드] {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL)
    return LOCAL_MODEL


def estimate_yaw(landmarks, w: int, h: int) -> float:
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    nose_x = nose.x * w
    left_x = left_ear.x * w
    right_x = right_ear.x * w
    face_w = right_x - left_x
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

        largest = max(result.face_landmarks, key=lambda lm: face_bbox_area(lm, w, h))
        yaw = estimate_yaw(largest, w, h)
        direction = "정면" if abs(yaw) <= YAW_THRESHOLD else "측면"

        return {"head_pose": direction, "head_pose_face_count": face_count}
    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return {"head_pose": "오류", "head_pose_face_count": -1}


def main(args):
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("[설치 필요] pip install mediapipe")
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
        base_options = mp_python.BaseOptions(model_asset_path=str(get_model_path()))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            min_face_detection_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)

        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="머리 방향 추출")):
            cached[row["video_id"]] = extract_head_pose(row["resolved_path"], detector)
            if (i + 1) % 100 == 0:
                save_ckpt(cached, CHECKPOINT)

        detector.close()
        save_ckpt(cached, CHECKPOINT)

    poses = pd.Series({k: v["head_pose"] for k, v in cached.items()})
    print(f"[결과] head_pose 분포:\n{poses.value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 머리 방향 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
