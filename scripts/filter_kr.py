"""
{CAT}_final_clean.csv → {CAT}_final_kr.csv

필터 기준 (channel_id 단위, 하나라도 해당하면 채널 전체 제거):
  1. channel_name에 외국어 문자 포함 (일본어, 러시아어, 아랍어, 태국어 등)
  2. default_language가 ko / en / en-US / en-GB 외

실행:
    python scripts/filter_kr.py
"""
import unicodedata
import pandas as pd
from pathlib import Path

CATEGORIES = ["FOOD", "SOCIETY", "VLOG"]

ALLOWED_LANGS = {"ko", "en", "en-US", "en-GB"}

FOREIGN_SCRIPTS = [
    "CJK", "HIRAGANA", "KATAKANA",
    "ARABIC", "CYRILLIC", "THAI",
    "DEVANAGARI", "HEBREW",
]


def has_foreign_script(name: str) -> bool:
    for ch in str(name):
        uname = unicodedata.name(ch, "")
        if any(s in uname for s in FOREIGN_SCRIPTS):
            return True
    return False


for cat in CATEGORIES:
    clean_p = Path(f"Data/{cat}/{cat}_final_clean.csv")
    out_p   = Path(f"Data/{cat}/{cat}_final_kr_clean.csv")

    if not clean_p.exists():
        print(f"[{cat}] final_clean 없음 — 건너뜀")
        continue

    df = pd.read_csv(clean_p, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")

    # 채널 단위 대표 정보 (channel_id별 첫 행)
    df_ch = df[["channel_id", "channel_name", "default_language"]].drop_duplicates("channel_id")

    # 제거 대상 channel_id 수집
    name_mask = df_ch["channel_name"].apply(has_foreign_script)
    lang_mask = ~df_ch["default_language"].fillna("").isin(ALLOWED_LANGS)
    remove_ids = set(df_ch[name_mask | lang_mask]["channel_id"])

    df_kr = df[~df["channel_id"].isin(remove_ids)].reset_index(drop=True)
    df_kr.to_csv(out_p, index=False, encoding="utf-8-sig")

    n_removed_ch  = len(remove_ids)
    n_kept_ch     = df_ch["channel_id"].nunique() - n_removed_ch
    n_name_only   = int((name_mask & ~lang_mask).sum())
    n_lang_only   = int((~name_mask & lang_mask).sum())
    n_both        = int((name_mask & lang_mask).sum())

    print(f"[{cat}] 저장 완료 → {out_p}")
    print(f"  전체 행(clean)  : {len(df)}")
    print(f"  전체 채널       : {df_ch['channel_id'].nunique()}")
    print(f"  제거 채널       : {n_removed_ch}  (이름만:{n_name_only} / 언어만:{n_lang_only} / 둘다:{n_both})")
    print(f"  유지 채널       : {n_kept_ch}")
    print(f"  final_kr 행     : {len(df_kr)}")
    print()
