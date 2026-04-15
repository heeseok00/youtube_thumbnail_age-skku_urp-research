"""
Data/data_age_sex/{CAT}_age_sex.csv  +  Data/{CAT}/{CAT}_clean.csv
 → channel_id 추가 + 컬럼명 통일
 → Data/{CAT}/{CAT}_age_sex.csv 저장

실행:
    python scripts/merge_age_sex.py
"""
import pandas as pd
from pathlib import Path

CATEGORIES = ["EDU", "FOOD", "HEALTH", "SOCIETY", "VLOG"]

RENAME = {
    "title":       "channel_name",
    "gender_F":    "female_pct",
    "gender_M":    "male_pct",
    "age_0_17":    "age_~17",
    "age_18_24":   "age_18~24",
    "age_25_34":   "age_25~34",
    "age_35_44":   "age_35~44",
    "age_45_54":   "age_45~54",
    "age_55_64":   "age_55~64",
    "age_65_plus": "age_65~",
}

OUTPUT_COLS = [
    "channel_name", "channel_id", "channel_link",
    "subscriberCount", "dailyViewCount",
    "female_pct", "male_pct",
    "age_~17", "age_18~24", "age_25~34",
    "age_35~44", "age_45~54", "age_55~64", "age_65~",
]

for cat in CATEGORIES:
    src   = Path(f"Data/data_age_sex/{cat}_age_sex.csv")
    clean = Path(f"Data/{cat}/{cat}_clean.csv")
    out   = Path(f"Data/{cat}/{cat}_age_sex.csv")

    if not src.exists():
        print(f"[{cat}] 소스 없음: {src}")
        continue
    if not clean.exists():
        print(f"[{cat}] clean 없음: {clean}")
        continue

    df_ag = pd.read_csv(src,   encoding="utf-8-sig", on_bad_lines="skip")
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df_cl = pd.read_csv(clean, encoding=enc)[["channel_id", "channel_link", "channel_name"]]
            break
        except (UnicodeDecodeError, KeyError):
            continue

    # channel_link 기준으로 channel_id 병합
    df = df_ag.merge(df_cl[["channel_id", "channel_link"]], on="channel_link", how="left")

    # 컬럼명 통일
    df = df.rename(columns=RENAME)

    # 출력 컬럼 순서 정렬 (없는 컬럼은 제외)
    cols = [c for c in OUTPUT_COLS if c in df.columns]
    df = df[cols]

    # 기존 파일이 있으면 백업
    if out.exists():
        bak = out.with_suffix(".csv.bak")
        out.rename(bak)
        print(f"[{cat}] 기존 파일 백업: {bak.name}")

    df.to_csv(out, index=False, encoding="utf-8-sig")

    # 결과 요약
    matched   = df["channel_id"].notna().sum()
    unmatched = df["channel_id"].isna().sum()
    gender_ok = df["female_pct"].notna().sum()
    print(f"[{cat}] 저장 완료 → {out}")
    print(f"       전체 {len(df)}행 | channel_id 매칭 {matched} | 미매칭 {unmatched}")
    print(f"       gender 값 있음: {gender_ok} | null: {len(df)-gender_ok}")
    print()
