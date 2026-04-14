import pandas as pd
from pathlib import Path

meta = pd.read_csv("Data/VLOG_meta.csv", encoding="utf-8-sig")
att = pd.read_csv("Data/VLOG_meta_attempted.csv", encoding="utf-8-sig")
channels_input = pd.read_csv(
    "Data/Data_0413/clean_1(channel_id_컬럼추가)/VLOG_clean_1.csv",
    encoding="utf-8-sig",
)

meta_channels = set(meta["channel_id"].astype(str))
att_channels = set(att["channel_id"].astype(str))
input_channels = set(channels_input["channel_id"].astype(str))

in_meta_not_att = meta_channels - att_channels
in_att_not_meta = att_channels - meta_channels
remaining = input_channels - att_channels

print("=== 동기화 검증 ===")
print(f"입력 채널 수          : {len(input_channels):,}개")
print(f"meta 수집 채널 수     : {len(meta_channels):,}개")
print(f"attempted 기록 채널 수: {len(att_channels):,}개")
print()
print(f"meta에 있는데 attempted에 없는 채널: {len(in_meta_not_att)}개  ← 0이어야 정상")
print(f"attempted에 있는데 meta에 없는 채널: {len(in_att_not_meta)}개  ← 0개 수집 채널들")
print()
print(f"아직 미처리 채널 수   : {len(remaining):,}개")
print()
if len(in_meta_not_att) == 0:
    print("동기화 완료")
else:
    print(f"동기화 미완료: {len(in_meta_not_att)}개 채널 누락")
    print(in_meta_not_att)
