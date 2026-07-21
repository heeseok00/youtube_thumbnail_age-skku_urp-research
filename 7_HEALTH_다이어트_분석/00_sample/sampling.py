"""HEALTH '다이어트' subcategory 균형 표본 로더 (공용 모듈).

모든 분석 단계(01~04)는 이 모듈의 load_diet_sample()을 import해서 표본을 얻는다.
별도 표본 CSV를 만들지 않고, 원본 v3 CSV에서 시드 고정 샘플링으로 매번 동일한
2,500행(34-/65+ 각 1,250개)을 재현한다.

주의: 원본 HEALTH_new_category_v3.csv의 내용/행 순서가 바뀌면 뽑히는 표본도
바뀐다. 분석 기간 동안 v3 파일은 수정하지 말 것.

사용 예 (각 단계 노트북에서):
    import sys
    sys.path.insert(0, "/home/urp_jwl/URP_backup/26-1_URP/7_HEALTH_다이어트_분석/00_sample")
    from sampling import load_diet_sample
    df = load_diet_sample()
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 26-1_URP
INPUT_CSV = BASE_DIR / "Data/HEALTH/HEALTH_new_category_v3.csv"

SUBCATEGORY = "다이어트"
N_PER_GROUP = 1250
SEED = 42

AGE_COLS = ["age_~17", "age_18~24", "age_25~34",
            "age_35~44", "age_45~54", "age_55~64", "age_65~"]
YOUNG_COLS = ["age_~17", "age_18~24", "age_25~34"]

KEEP_COLS = ["video_id", "age_group", "channel_id", "channel_name",
             "title", "thumbnail_path", "subcategory"] + AGE_COLS


def load_diet_sample(resolve_thumbnails=True):
    """시드 고정 균형 표본 2,500행 반환 (34-/65+ 각 1,250개).

    resolve_thumbnails=True면 'resolved_path' 컬럼(절대경로)을 추가하고
    파일이 실제 존재하는 행만 남긴다.
    """
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["subcategory"] == SUBCATEGORY].dropna(subset=AGE_COLS).copy()

    # 영상별 dominant 연령대 -> 이진 그룹 (35~64 dominant는 제외)
    dominant = df[AGE_COLS].idxmax(axis=1)
    df["age_group"] = None
    df.loc[dominant.isin(YOUNG_COLS), "age_group"] = "34-"
    df.loc[dominant == "age_65~", "age_group"] = "65+"
    df = df[df["age_group"].notna()]

    sample = (
        df.groupby("age_group", group_keys=False)
          .apply(lambda g: g.sample(n=N_PER_GROUP, random_state=SEED))
          .reset_index(drop=True)
    )[KEEP_COLS]

    if resolve_thumbnails:
        sample["resolved_path"] = sample["thumbnail_path"].apply(
            lambda p: str(BASE_DIR / str(p).replace("\\", "/"))
        )
        exists = sample["resolved_path"].apply(lambda p: Path(p).exists())
        if exists.sum() < len(sample):
            print(f"[경고] 썸네일 누락 {len(sample) - exists.sum()}개 행 제외")
        sample = sample[exists].reset_index(drop=True)

    return sample


if __name__ == "__main__":
    s = load_diet_sample()
    print(f"표본 {len(s):,}행 / {s['age_group'].value_counts().to_dict()}")
    print(f"채널 수: 34-={s[s.age_group=='34-'].channel_id.nunique()}, "
          f"65+={s[s.age_group=='65+'].channel_id.nunique()}")
