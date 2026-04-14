import pandas as pd
from pathlib import Path

data_dir = Path("Data/Data_0413")
files = ["EDU.csv", "FOOD.csv", "HEALTH.csv", "SOCIETY.csv", "VLOG.csv"]

for fname in files:
    src = data_dir / fname
    try:
        df = pd.read_csv(src, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(src, encoding="cp949")
    print(f"{fname}: {list(df.columns)} | {len(df)} rows")

    df["channel_id"] = df["channel_link"].str.extract(r"(UC[\w-]+)")

    cols = list(df.columns)
    cols.remove("channel_id")
    name_idx = cols.index("channel_name")
    cols.insert(name_idx + 1, "channel_id")
    df = df[cols]

    stem = fname.replace(".csv", "")
    out = data_dir / f"{stem}_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    sample = df["channel_id"].iloc[0]
    print(f"  -> {out.name} 저장 완료 | channel_id 샘플: {sample}")
