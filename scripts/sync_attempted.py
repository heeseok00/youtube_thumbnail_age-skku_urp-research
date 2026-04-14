import csv
import datetime
import pandas as pd
from pathlib import Path

meta_path = Path("Data/VLOG_meta.csv")
attempted_path = Path("Data/VLOG_meta_attempted.csv")

meta = pd.read_csv(meta_path, encoding="utf-8-sig")
counts = meta.groupby("channel_id")["video_id"].count().reset_index()
counts.columns = ["channel_id", "saved_count"]

if attempted_path.exists():
    att = pd.read_csv(attempted_path, encoding="utf-8-sig")
    already = set(att["channel_id"].astype(str))
    print(f"기존 attempted 채널: {len(already)}개")
else:
    already = set()
    print("attempted.csv 없음 → 새로 생성")

to_add = counts[~counts["channel_id"].astype(str).isin(already)]
print(f"추가할 채널: {len(to_add)}개")

if len(to_add) == 0:
    print("추가할 내용 없음")
else:
    today = datetime.date.today().isoformat()
    write_header = not attempted_path.exists()
    with attempted_path.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=["channel_id", "saved_count", "attempted_at"])
        if write_header:
            writer.writeheader()
        for _, row in to_add.iterrows():
            writer.writerow({
                "channel_id": row["channel_id"],
                "saved_count": int(row["saved_count"]),
                "attempted_at": today,
            })

    print(f"완료: {attempted_path} → 총 {len(already) + len(to_add)}개 채널 기록")
