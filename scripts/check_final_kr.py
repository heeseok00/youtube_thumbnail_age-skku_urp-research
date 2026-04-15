import unicodedata
import pandas as pd
from pathlib import Path

CATEGORIES = ["FOOD", "SOCIETY", "VLOG"]
FOREIGN_SCRIPTS = ["CJK","HIRAGANA","KATAKANA","ARABIC","CYRILLIC","THAI","DEVANAGARI","HEBREW"]
ALLOWED_LANGS = {"ko", "en", "en-US", "en-GB"}

def has_foreign_script(name):
    for ch in str(name):
        if any(s in unicodedata.name(ch, "") for s in FOREIGN_SCRIPTS):
            return True
    return False

for cat in CATEGORIES:
    p = Path(f"Data/{cat}/{cat}_final_kr.csv")
    if not p.exists():
        print(f"[{cat}] 파일 없음")
        continue

    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")
    df_ch = df[["channel_id","channel_name","default_language"]].drop_duplicates("channel_id")

    print(f"{'='*50}")
    print(f"[{cat}] {p}")
    print(f"{'='*50}")

    # 기본 통계
    print(f"총 행(영상)     : {len(df)}")
    print(f"고유 채널       : {len(df_ch)}")
    vc = df.groupby("channel_id").size()
    print(f"영상/채널: min={vc.min()}  max={vc.max()}  median={int(vc.median())}")

    # 핵심 컬럼 null
    for col in ["video_id","title","thumbnail_path","female_pct","male_pct","age_~17"]:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                print(f"  null {col}: {n}")

    # 썸네일 파일 존재 확인
    tp = df["thumbnail_path"].fillna("").astype(str).str.strip()
    tp_valid = tp[(tp != "") & (tp != "nan")]
    exist = sum(1 for p2 in tp_valid if Path(p2).exists())
    missing = len(tp_valid) - exist
    print(f"thumbnail 파일 존재: {exist} / {len(tp_valid)}  (없음: {missing})")

    # 필터 잔존 여부 재검증
    foreign_name = df_ch["channel_name"].apply(has_foreign_script).sum()
    foreign_lang = (~df_ch["default_language"].fillna("").isin(ALLOWED_LANGS)).sum()
    print(f"외국어 채널명 잔존: {foreign_name}  (0이어야 정상)")
    print(f"외국어 언어코드 잔존: {foreign_lang}  (0이어야 정상)")

    # default_language 분포
    print(f"default_language 분포: {df_ch['default_language'].value_counts().to_dict()}")

    # 중복 video_id
    dup = int(df["video_id"].duplicated().sum())
    print(f"video_id 중복: {dup}")

    print()
