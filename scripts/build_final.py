"""
{CAT}_meta.csv  +  {CAT}_age_sex.csv  →  {CAT}_final.csv

- channel_id 기준 left join (meta 기준 유지)
- age_sex에서 채널 단위 컬럼만 붙임 (channel_name, channel_id 중복 제거)

실행:
    python scripts/build_final.py
"""
import pandas as pd
from pathlib import Path

CATEGORIES = ["FOOD", "SOCIETY", "VLOG"]

AGE_SEX_COLS = [
    "channel_link", "subscriberCount", "dailyViewCount",
    "female_pct", "male_pct",
    "age_~17", "age_18~24", "age_25~34",
    "age_35~44", "age_45~54", "age_55~64", "age_65~",
]

for cat in CATEGORIES:
    meta_p = Path(f"Data/{cat}/{cat}_meta.csv")
    age_p  = Path(f"Data/{cat}/{cat}_age_sex.csv")
    out_p  = Path(f"Data/{cat}/{cat}_final.csv")

    if not meta_p.exists():
        print(f"[{cat}] meta 없음 — 건너뜀")
        continue
    if not age_p.exists():
        print(f"[{cat}] age_sex 없음 — 건너뜀")
        continue

    df_meta = pd.read_csv(meta_p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    df_age  = pd.read_csv(age_p,  encoding="utf-8-sig", on_bad_lines="skip")

    # age_sex에서 channel_id + 필요 컬럼만 추출 (중복 channel_name 제거)
    cols_to_add = ["channel_id"] + [c for c in AGE_SEX_COLS if c in df_age.columns]
    df_age_slim = df_age[cols_to_add].drop_duplicates(subset="channel_id")

    # left join: meta 행 수 유지, 채널에 age/sex 정보 붙임
    df_final = df_meta.merge(df_age_slim, on="channel_id", how="left")

    df_final.to_csv(out_p, index=False, encoding="utf-8-sig")

    # 요약
    total      = len(df_final)
    ch_total   = df_final["channel_id"].nunique()
    ch_age_ok  = df_final[df_final["female_pct"].notna()]["channel_id"].nunique()
    row_age_ok = df_final["female_pct"].notna().sum()

    print(f"[{cat}] 저장 완료 → {out_p}")
    print(f"  영상(행) 수     : {total}")
    print(f"  고유 채널       : {ch_total}")
    print(f"  age/sex 있는 채널: {ch_age_ok} / {ch_total}")
    print(f"  age/sex 있는 행 : {row_age_ok} / {total}")
    print(f"  컬럼 수         : {len(df_final.columns)}")
    print(f"  컬럼 목록       : {list(df_final.columns)}")
    print()
