"""
4단계(시각 피처) 공통 유틸리티
- 다이어트 샘플 로딩 (00_sample/sampling.py 공유 모듈)
- 딕셔너리 기반 체크포인트 저장/로드
"""

import sys
from pathlib import Path

import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent           # 04_visual_features
PROJECT_DIR = STAGE_DIR.parent                        # 7_HEALTH_다이어트_분석
CKPT_DIR = STAGE_DIR / "checkpoints"
OUT_DIR = STAGE_DIR / "outputs"

sys.path.insert(0, str(PROJECT_DIR / "00_sample"))
from sampling import load_diet_sample  # noqa: E402


def load_sample() -> pd.DataFrame:
    """고정 시드 다이어트 샘플 (thumbnail 존재 행만, resolved_path 포함)."""
    return load_diet_sample(resolve_thumbnails=True)


def load_ckpt(path: Path, cols: list[str]) -> dict:
    """체크포인트 CSV → {video_id: {col: val, ...}}"""
    if not path.exists():
        return {}
    ckpt_df = pd.read_csv(path)
    return {
        row["video_id"]: {c: row.get(c) for c in cols}
        for _, row in ckpt_df.iterrows()
    }


def save_ckpt(cached: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"video_id": k, **v} for k, v in cached.items()]).to_csv(path, index=False)
