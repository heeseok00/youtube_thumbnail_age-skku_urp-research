import os
import pandas as pd

meta = pd.read_csv("Data/VLOG_meta.csv", encoding="utf-8-sig")
channels_input = pd.read_csv(
    "Data/Data_0413/clean_1(channel_id_컬럼추가)/VLOG_clean_1.csv",
    encoding="utf-8-sig",
)

print("=== VLOG_meta.csv ===")
print(f"총 영상 수         : {len(meta):,}개")
print(f"수집된 채널 수     : {meta['channel_id'].nunique():,}개")
print(f"입력 채널 수       : {len(channels_input):,}개")

counts = meta.groupby("channel_id")["video_id"].count()
print()
print("=== 채널당 영상 수 분포 ===")
print(f"10개 완료 채널     : {(counts == 10).sum():,}개")
print(f"1~9개 채널         : {((counts >= 1) & (counts < 10)).sum():,}개")
print(f"평균 영상 수       : {counts.mean():.2f}개")

collected_ids = set(meta["channel_id"].astype(str))
input_ids = set(channels_input["channel_id"].astype(str))
not_collected = input_ids - collected_ids
print()
print("=== 미수집 채널 ===")
print(f"미수집 채널 수     : {len(not_collected):,}개")

attempted_path = "Data/VLOG_meta_attempted.csv"
if os.path.exists(attempted_path):
    att = pd.read_csv(attempted_path, encoding="utf-8-sig")
    att_ids = set(att["channel_id"].astype(str))
    truly_remaining = not_collected - att_ids
    print()
    print("=== VLOG_meta_attempted.csv ===")
    print(f"시도 완료 채널     : {len(att):,}개")
    print(f"저장 0개 채널      : {(att['saved_count'] == 0).sum():,}개")
    print(f"저장 1~9개 채널    : {((att['saved_count'] >= 1) & (att['saved_count'] < 10)).sum():,}개")
    print()
    print("=== 최종 미처리 채널 ===")
    print(f"(미수집 - 시도완료): {len(truly_remaining):,}개")
else:
    print("(attempted.csv 없음 - 아직 이번 버전으로 실행 안됨)")
