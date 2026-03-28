"""
YT_ChannelData_Health_clean.csv 의 구독자 수 기준으로
파이프라인용 채널 레이블 CSV를 생성합니다.

  - small  : subscribers < 50,000
  - medium : 50,000 ≤ subscribers < 200,000
  - large  : subscribers ≥ 200,000
"""
import pandas as pd
from pathlib import Path

base = Path(__file__).parent

health_ds   = pd.read_csv(base / "YT_dataset_health.csv",        encoding="utf-8-sig")
health_ch   = pd.read_csv(base / "YT_ChannelData_Health_clean.csv", encoding="utf-8-sig")

# 영상 데이터에 있는 channel_id 목록
video_channel_ids = set(health_ds["channel_id"].astype(str))

# subscribers 컬럼으로 티어 생성
def tier(subs):
    if subs < 50_000:
        return "small"
    elif subs < 200_000:
        return "medium"
    else:
        return "large"

health_ch = health_ch[health_ch["channel_id"].astype(str).isin(video_channel_ids)].copy()
health_ch["target_age"] = health_ch["subscribers"].apply(tier)

out = health_ch[["channel_name", "channel_id", "target_age"]].drop_duplicates("channel_id")
out.to_csv(base / "YT_channelsList_health.csv", index=False, encoding="utf-8-sig")

print(f"저장 완료: YT_channelsList_health.csv  ({len(out)}개 채널)")
print(out.groupby("target_age").size().rename("채널 수").to_string())

# 영상 데이터 coverage 확인
matched = video_channel_ids & set(out["channel_id"].astype(str))
print(f"\n영상 데이터 커버: {len(matched)} / {len(video_channel_ids)} channel_id")
