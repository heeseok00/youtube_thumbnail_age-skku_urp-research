"""
02_text_ratio.py
다이어트 샘플 썸네일에서 '텍스트가 차지하는 면적 비율'을 추출 (EasyOCR).
(원본: 5_썸네일 피처 분석 파이프라인/02_text_ratio.py — 데이터 로딩만 변경)

출력 컬럼: text_ratio (0.0 ~ 1.0)
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_sample, save_ckpt, CKPT_DIR

OCR_TIMEOUT = 30  # 이미지당 최대 30초
CHECKPOINT = CKPT_DIR / "02_text_ratio.csv"
COL = "text_ratio"


def _persistent_worker(in_q, out_q):
    """모델을 한 번만 로딩하고 요청마다 OCR 실행하는 워커 프로세스."""
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=True, verbose=False)
    while True:
        img_path = in_q.get()
        if img_path is None:
            break
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            total_pixels = w * h
            results = ocr.readtext(img_path)
            if not results:
                out_q.put(0.0)
                continue

            def polygon_area(pts):
                n = len(pts)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += pts[i][0] * pts[j][1]
                    area -= pts[j][0] * pts[i][1]
                return abs(area) / 2.0

            text_pixels = sum(polygon_area(box) for box, _, _ in results)
            out_q.put(min(text_pixels / total_pixels, 1.0))
        except Exception:
            out_q.put(float("nan"))


class PersistentOCR:
    """워커 프로세스를 재사용하며 이미지당 timeout을 보장하는 OCR 래퍼."""

    def __init__(self):
        self._start_worker()

    def _start_worker(self):
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.worker = mp.Process(target=_persistent_worker,
                                 args=(self.in_q, self.out_q), daemon=True)
        self.worker.start()

    def readtext_ratio(self, img_path: str) -> float:
        self.in_q.put(img_path)
        try:
            return self.out_q.get(timeout=OCR_TIMEOUT)
        except Exception:
            print(f"\n[타임아웃] {img_path} — 워커 재시작")
            self.worker.terminate()
            self.worker.join()
            self._start_worker()
            return float("nan")

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
        print("[설치 필요] pip install easyocr")
        sys.exit(1)

    target = load_sample()
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] 다이어트 샘플: {len(target):,}개")

    cached = {}
    if CHECKPOINT.exists():
        ckpt_df = pd.read_csv(CHECKPOINT)
        cached = dict(zip(ckpt_df["video_id"], ckpt_df[COL]))
        cached = {k: v for k, v in cached.items() if not pd.isna(v)}
        print(f"[체크포인트] {len(cached):,}건 불러옴")

    remaining = target[~target["video_id"].isin(cached.keys())]
    print(f"[남은 작업] {len(remaining):,}건")

    if len(remaining) > 0:
        ocr = PersistentOCR()
        try:
            for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="텍스트 비중 추출")):
                cached[row["video_id"]] = ocr.readtext_ratio(row["resolved_path"])
                if (i + 1) % 100 == 0:
                    save_ckpt({k: {COL: v} for k, v in cached.items()}, CHECKPOINT)
        finally:
            ocr.close()
        save_ckpt({k: {COL: v} for k, v in cached.items()}, CHECKPOINT)

    vals = pd.Series(cached)
    print(f"[결과] {COL} 완료: {vals.notna().sum():,}건 / 평균 {vals.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 썸네일 텍스트 비중 추출")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
