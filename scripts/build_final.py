"""
{CAT}_meta.csv  +  {CAT}_age_sex.csv
  → {CAT}_final_raw.csv   : 전체 merge 결과 (결측 유지)
  → {CAT}_final_clean.csv : 핵심 컬럼 결측 행 제거

핵심 컬럼 (이 중 하나라도 null이면 clean에서 제외):
    video_id, title, thumbnail_path, female_pct, male_pct, age_~17

실행:
    python scripts/build_final.py
"""
import pandas as pd
from pathlib import Path

CATEGORIES = ["FOOD", "SOCIETY", "VLOG", "HEALTH"]

# 썸네일 다운로드를 나중에 따로 하는 카테고리 → thumbnail_path 파일 실존 체크 스킵
NO_THUMBNAIL_CHECK_CATS = {"HEALTH"}

AGE_SEX_COLS = [
    "channel_link", "subscriberCount", "dailyViewCount",
    "female_pct", "male_pct",
    "age_~17", "age_18~24", "age_25~34",
    "age_35~44", "age_45~54", "age_55~64", "age_65~",
]

# clean에서 결측을 허용하지 않는 핵심 컬럼
REQUIRED_COLS = ["video_id", "title", "thumbnail_path", "female_pct", "male_pct", "age_~17"]

for cat in CATEGORIES:
    meta_p      = Path(f"Data/{cat}/{cat}_meta.csv")
    age_p       = Path(f"Data/{cat}/{cat}_age_sex.csv")
    raw_out_p   = Path(f"Data/{cat}/{cat}_final_raw.csv")
    clean_out_p = Path(f"Data/{cat}/{cat}_final_clean.csv")

    if not meta_p.exists():
        print(f"[{cat}] meta 없음 — 건너뜀")
        continue
    if not age_p.exists():
        print(f"[{cat}] age_sex 없음 — 건너뜀")
        continue

    df_meta = pd.read_csv(meta_p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    df_age  = pd.read_csv(age_p,  encoding="utf-8-sig", on_bad_lines="skip")

    cols_to_add = ["channel_id"] + [c for c in AGE_SEX_COLS if c in df_age.columns]
    df_age_slim = df_age[cols_to_add].drop_duplicates(subset="channel_id")

    df_raw = df_meta.merge(df_age_slim, on="channel_id", how="left")

    # ── final_raw: 전체 보존 ────────────────────────────────
    df_raw.to_csv(raw_out_p, index=False, encoding="utf-8-sig")

    # ── final_clean: 핵심 컬럼 결측 행 제거 + 썸네일 파일 실존 확인 ──
    # thumbnail_path 체크 스킵 카테고리는 required에서 제외
    skip_thumb_check = cat in NO_THUMBNAIL_CHECK_CATS
    req_cols = [c for c in REQUIRED_COLS if c != "thumbnail_path"] if skip_thumb_check else REQUIRED_COLS
    req_present = [c for c in req_cols if c in df_raw.columns]
    df_clean = df_raw.dropna(subset=req_present).reset_index(drop=True)

    # thumbnail_path가 있지만 실제 파일이 없는 행 제거 (스킵 카테고리는 건너뜀)
    if not skip_thumb_check and "thumbnail_path" in df_clean.columns:
        file_exists = df_clean["thumbnail_path"].apply(lambda p: Path(str(p)).exists())
        dropped_no_file = int((~file_exists).sum())
        df_clean = df_clean[file_exists].reset_index(drop=True)
        if dropped_no_file:
            print(f"  파일 없는 행 추가 제거: {dropped_no_file}행")
    elif skip_thumb_check:
        print(f"  [참고] thumbnail_path 파일 실존 체크 스킵 ({cat}: 썸네일 나중에 다운로드)")

    df_clean.to_csv(clean_out_p, index=False, encoding="utf-8-sig")

    # 요약
    dropped = len(df_raw) - len(df_clean)
    print(f"[{cat}]")
    print(f"  final_raw.csv   : {len(df_raw)}행  → {raw_out_p}")
    print(f"  final_clean.csv : {len(df_clean)}행  → {clean_out_p}")
    print(f"  제거된 행       : {dropped}행  (기준: {req_present})")
    print(f"  고유 채널 (raw) : {df_raw['channel_id'].nunique()}")
    print(f"  고유 채널 (clean): {df_clean['channel_id'].nunique()}")
    print()

# 기존 _final.csv 는 _final_raw.csv 로 대체됐으므로 안내만 출력
print("※ 기존 *_final.csv 파일은 *_final_raw.csv 로 대체되었습니다.")
print("  필요 없으면 직접 삭제하세요.")
