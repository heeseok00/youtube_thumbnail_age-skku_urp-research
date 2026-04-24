#!/usr/bin/env python3
"""Download thumbnails from a metadata CSV and update thumbnail_path column.

Usage:
    python data_pipeline/03_thumbnail_download/03.thumbnail_download.py --meta-csv Data/VLOG/VLOG_meta.csv
    python data_pipeline/03_thumbnail_download/03.thumbnail_download.py --meta-csv Data/VLOG/VLOG_meta.csv --workers 16

By default thumbnails are saved to  <meta_csv_dir>/thumbnails/<channel_name>/<video_id>.jpg
e.g. data/VLOG/thumbnails/ChannelName/abc123.jpg
"""
from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

DEFAULT_WORKERS = 64
DEFAULT_TIMEOUT = 5
DEFAULT_RETRIES = 2

_thread_local = threading.local()


# ── 유틸리티 ────────────────────────────────────────────────

def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def sanitize(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    return safe or "unknown_channel"


def derive_category(meta_csv: Path) -> str:
    return meta_csv.stem.split("_")[0].upper()


def get_session(user_agent: str) -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": user_agent})
        _thread_local.session = sess
    return sess


# ── 다운로드 ────────────────────────────────────────────────

def download_one(
    url: str,
    dest: Path,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> bool:
    if not url:
        return False
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            resp = get_session(user_agent).get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(0.3 * (2 ** attempt))
    return False


# ── 메인 로직 ───────────────────────────────────────────────

def download_thumbnails(
    meta_csv: Path,
    thumbnail_dir: Path,
    workers: int,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> None:
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")

    required = {"channel_name", "video_id", "thumbnail_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼 없음: {sorted(missing)}")

    if "thumbnail_path" not in df.columns:
        df["thumbnail_path"] = ""
    df["thumbnail_path"] = df["thumbnail_path"].fillna("").astype(str)

    category = derive_category(meta_csv)
    base_dir = thumbnail_dir

    # 다운로드 대상: thumbnail_url 있고 파일이 아직 없는 행
    def needs_download(row: pd.Series) -> bool:
        url = str(row.get("thumbnail_url", "") or "").strip()
        if not url:
            return False
        path_str = str(row.get("thumbnail_path", "") or "").strip()
        if path_str and Path(path_str).exists():
            return False
        return True

    targets = df[df.apply(needs_download, axis=1)].copy()
    already = len(df) - len(targets)

    print(f"\n[{category}] 전체 {len(df)}개 | 이미 완료 {already}개 | 다운로드 대상 {len(targets)}개")

    if targets.empty:
        print(f"[{category}] 모두 완료됨.")
        return

    # 작업 목록 생성
    jobs: list[tuple[int, str, Path]] = []
    for idx, row in targets.iterrows():
        url = str(row["thumbnail_url"]).strip()
        channel_safe = sanitize(str(row["channel_name"]))
        dest = base_dir / channel_safe / f"{row['video_id']}.jpg"
        jobs.append((idx, url, dest))

    success = 0
    fail = 0
    lock = threading.Lock()
    started_at = time.monotonic()
    SAVE_INTERVAL = 500
    LOG_INTERVAL = 100

    def save_csv() -> None:
        df.to_csv(meta_csv, index=False, encoding="utf-8-sig")

    def task(job: tuple[int, str, Path]) -> tuple[int, Path, bool]:
        idx, url, dest = job
        ok = download_one(url, dest, user_agent, timeout=timeout, retries=retries)
        return idx, dest, ok

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task, j): j for j in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                idx, dest, ok = future.result()
                with lock:
                    if ok:
                        df.at[idx, "thumbnail_path"] = str(dest)
                        success += 1
                    else:
                        fail += 1

                    if i % LOG_INTERVAL == 0 or i == len(jobs):
                        elapsed = time.monotonic() - started_at
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (len(jobs) - i) / rate if rate > 0 else 0
                        print(
                            f"  {i}/{len(jobs)} | 성공 {success} 실패 {fail} | "
                            f"경과 {format_seconds(elapsed)} | 예상 남은 {format_seconds(eta)}"
                        )
                    if i % SAVE_INTERVAL == 0 or i == len(jobs):
                        save_csv()
    except KeyboardInterrupt:
        print(f"\n[{category}] 중단됨 — 진행분 저장 중...")
        save_csv()
        print(f"[{category}] 저장 완료: 성공 {success}개 / 실패 {fail}개 → {meta_csv}")
        return

    print(f"\n[{category}] 완료: 성공 {success}개 / 실패 {fail}개 → {meta_csv} 업데이트")


# ── CLI ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="메타데이터 CSV의 thumbnail_url로 썸네일 다운로드")
    parser.add_argument(
        "--meta-csv",
        type=Path,
        required=True,
        help="메타데이터 CSV 경로 (예: Data/VLOG_meta.csv)",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=None,
        help="썸네일 저장 디렉터리 (기본값: <meta-csv 위치>/thumbnails/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"병렬 다운로드 스레드 수 (기본값: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0",
        help="HTTP User-Agent",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"요청 타임아웃 초 (기본값: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"실패 시 재시도 횟수 (기본값: {DEFAULT_RETRIES})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.meta_csv.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.meta_csv}")

    thumbnail_dir = args.thumbnail_dir if args.thumbnail_dir else args.meta_csv.parent / "thumbnails"

    download_thumbnails(
        meta_csv=args.meta_csv,
        thumbnail_dir=thumbnail_dir,
        workers=max(1, args.workers),
        user_agent=args.user_agent,
        timeout=args.timeout,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
