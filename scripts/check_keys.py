"""YouTube API 키 quota 상태 확인"""
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

keys = []
i = 1
while True:
    k = os.getenv(f"YOUTUBE_API_KEY_{i}")
    if not k:
        break
    keys.append((i, k))
    i += 1

print(f"총 {len(keys)}개 키 확인\n")

ok_count = 0
for idx, key in keys:
    masked = key[:8] + "..." + key[-4:]
    try:
        yt = build("youtube", "v3", developerKey=key)
        yt.videos().list(part="snippet", id="dQw4w9WgXcQ").execute()
        print(f"  KEY_{idx} ({masked})  ✅ 사용 가능")
        ok_count += 1
    except HttpError as e:
        reason = ""
        if e.resp.status == 403:
            import json
            try:
                err = json.loads(e.content)["error"]["errors"][0]["reason"]
                reason = err
            except Exception:
                reason = "403 Forbidden"
        if reason in ("quotaExceeded", "dailyLimitExceeded"):
            print(f"  KEY_{idx} ({masked})  ❌ 할당량 초과")
        else:
            print(f"  KEY_{idx} ({masked})  ⚠️  오류: {reason or e.resp.status}")
    except Exception as e:
        print(f"  KEY_{idx} ({masked})  ⚠️  예외: {e}")

print(f"\n결과: {ok_count}/{len(keys)}개 사용 가능")
