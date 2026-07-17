"""
02_text_ratio.py
썸네일에서 '텍스트가 차지하는 면적 비율'을 추출합니다.

의존성: easyocr (이미 설치됨)
    pip install easyocr

출력 컬럼: text_ratio (0.0 ~ 1.0)
"""

import argparse
import sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OCR_TIMEOUT = 30  # 이미지당 최대 30초

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_csv, save_csv, resolve_thumbnail_path,
    get_target_rows, save_checkpoint, resolve_missing
)

CHECKPOINT = Path(__file__).parent / "checkpoints/02_text_ratio.csv"
COL = "text_ratio"


def _persistent_worker(in_q, out_q):
    """모델을 한 번만 로딩하고 요청마다 OCR 실행하는 워커 프로세스."""
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=True, verbose=False)
    while True:
        img_path = in_q.get()
        if img_path is None:  # 종료 신호
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
        self.in_q  = mp.Queue()
        self.out_q = mp.Queue()
        self.worker = mp.Process(target=_persistent_worker,
                                 args=(self.in_q, self.out_q), daemon=True)
        self.worker.start()

    def readtext_ratio(self, img_path: str) -> float:
        self.in_q.put(img_path)
        try:
            return self.out_q.get(timeout=OCR_TIMEOUT)
        except Exception:
            # 타임아웃 → 워커 kill 후 재시작
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


def extract_text_ratio(img_path: str, ocr: "PersistentOCR") -> float:
    """PersistentOCR 래퍼를 통해 텍스트 면적 비율 계산."""
    try:
        return ocr.readtext_ratio(img_path)
    except Exception as e:
        print(f"[오류] {img_path}: {e}")
        return float("nan")


def main(args):
    try:
        import easyocr  # noqa: F401
    except ImportError:
        print("[설치 필요] pip install easyocr")
        sys.exit(1)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    target = get_target_rows(df, args.subcategory)
    if args.test > 0:
        target = target.head(args.test)
        print(f"[테스트 모드] {args.test}개만 처리")
    print(f"[대상] '{args.subcategory}' 영상 수: {len(target):,}개")

    remaining = resolve_missing(target, COL, str(CHECKPOINT))
    results = dict(zip(target["video_id"], target.get(COL, [float("nan")] * len(target))))
    already_done = {k: v for k, v in results.items() if not pd.isna(v)}

    if len(remaining) == 0:
        print("[완료] 모든 영상 처리됨")
    else:
        ocr = PersistentOCR()
        try:
            for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="텍스트 비중 추출")):
                img_path = resolve_thumbnail_path(row["thumbnail_path"])
                ratio = extract_text_ratio(str(img_path), ocr)
                already_done[row["video_id"]] = ratio

                if (i + 1) % 100 == 0:
                    save_checkpoint(already_done, str(CHECKPOINT), COL)
        finally:
            ocr.close()

        save_checkpoint(already_done, str(CHECKPOINT), COL)

    df[COL] = df["video_id"].map(already_done)
    output_path = args.output or args.input.replace(".csv", f"_{COL}.csv")
    save_csv(df, output_path)
    print(f"[결과] {COL} 비null 수: {df[COL].notna().sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="썸네일 텍스트 비중 추출")
    parser.add_argument("--input", default=str(
        Path(__file__).parent.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--test", type=int, default=0, help="테스트: N개만 처리 (0=전체)")
    main(parser.parse_args())
