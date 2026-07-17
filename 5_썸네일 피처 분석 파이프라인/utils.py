"""
공통 유틸리티 모듈
- 썸네일 경로 변환 (Windows → Linux)
- 체크포인트 관리
- CSV 로드/저장
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/urp_jwl2/26-1_URP")
DEFAULT_CSV = BASE_DIR / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv"


def resolve_thumbnail_path(thumbnail_path: str) -> Path:
    """CSV에 저장된 Windows 스타일 경로를 Linux 절대 경로로 변환."""
    normalized = str(thumbnail_path).replace("\\", "/")
    return BASE_DIR / normalized


def load_csv(csv_path: str = None) -> pd.DataFrame:
    path = csv_path or DEFAULT_CSV
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"[저장 완료] {output_path} ({len(df):,}행)")


def load_checkpoint(checkpoint_path: str) -> dict:
    """체크포인트 파일 로드 (video_id → 결과값 딕셔너리)."""
    if os.path.exists(checkpoint_path):
        ckpt_df = pd.read_csv(checkpoint_path)
        return dict(zip(ckpt_df["video_id"], ckpt_df.iloc[:, 1]))
    return {}


def save_checkpoint(results: dict, checkpoint_path: str, col_name: str):
    """현재까지의 결과를 체크포인트 파일로 저장."""
    pd.DataFrame({
        "video_id": list(results.keys()),
        col_name: list(results.values())
    }).to_csv(checkpoint_path, index=False)


def get_target_rows(df: pd.DataFrame, subcategory: str = "시사/뉴스/사건") -> pd.DataFrame:
    """특정 subcategory 필터링."""
    return df[df["subcategory"] == subcategory].copy()


def resolve_missing(df: pd.DataFrame, col_name: str, checkpoint_path: str) -> pd.DataFrame:
    """체크포인트에서 기존 결과를 불러와 DataFrame에 병합, 미처리 행만 반환."""
    cached = load_checkpoint(checkpoint_path)
    if cached:
        df[col_name] = df["video_id"].map(cached)
        remaining = df[df[col_name].isna()]
        print(f"[체크포인트] {len(cached):,}건 불러옴 / 남은 작업: {len(remaining):,}건")
        return remaining
    return df
