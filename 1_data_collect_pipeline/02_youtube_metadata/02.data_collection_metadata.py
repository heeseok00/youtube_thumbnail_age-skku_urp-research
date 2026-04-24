#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


DEFAULT_CHANNELS_CSV = Path("YT_ChannelData_Health_clean.csv")
DEFAULT_OUTPUT_CSV = Path("YT_dataset_v1.csv")
DEFAULT_THUMBNAIL_DIR = Path("thumbnails")
DEFAULT_PLAYLIST_PAGE_SIZE = 50
DEFAULT_THUMBNAIL_WORKERS = 8
DEFAULT_SHORT_DURATION_SEC = 180
SHORT_KEYWORD_RE = re.compile(r"(?i)(?:^|\s|#)(shorts?|쇼츠)\b")

FIELDNAMES = [
    "channel_name",
    "channel_id",
    "video_id",
    "title",
    "description",
    "published_at",
    "tags",
    "category_id",
    "default_language",
    "duration",
    "dimension",
    "definition",
    "caption",
    "view_count",
    "like_count",
    "favorite_count",
    "comment_count",
    "privacy_status",
    "license",
    "embeddable",
    "made_for_kids",
    "topic_categories",
    "thumbnail_url",
    "thumbnail_path",
]

_thread_local = threading.local()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"

    filled = min(width, int(width * (completed / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class ProgressTracker:
    def __init__(self, total_channels: int) -> None:
        self.total_channels = total_channels
        self.started_at = time.monotonic()
        self.completed_channels = 0
        self.total_saved = 0

    def start_message(self, current_channel_name: str) -> str:
        current_index = self.completed_channels + 1
        bar = render_progress_bar(self.completed_channels, self.total_channels)
        return (
            f"{bar} 채널 {current_index}/{self.total_channels} 시작 | "
            f"현재 채널: {current_channel_name}"
        )

    def finish_message(self, channel_name: str, saved_count: int) -> str:
        self.completed_channels += 1
        self.total_saved += saved_count

        elapsed = time.monotonic() - self.started_at
        avg_per_channel = elapsed / self.completed_channels if self.completed_channels else None
        remaining_channels = self.total_channels - self.completed_channels
        eta = avg_per_channel * remaining_channels if avg_per_channel is not None else None
        bar = render_progress_bar(self.completed_channels, self.total_channels)

        return (
            f"{bar} 채널 {self.completed_channels}/{self.total_channels} 완료 | "
            f"최근 채널: {channel_name} | "
            f"누적 저장: {self.total_saved}개 | "
            f"경과: {format_seconds(elapsed)} | "
            f"예상 남은 시간: {format_seconds(eta)}"
        )


def get_api_keys(cli_value: str | None) -> list[str]:
    load_dotenv()
    if cli_value:
        return [cli_value]

    keys: list[str] = []
    i = 1
    while True:
        key = os.getenv(f"YOUTUBE_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1

    if not keys:
        single = os.getenv("YOUTUBE_API_KEY")
        if single:
            keys.append(single)

    if not keys:
        raise ValueError(
            "YOUTUBE_API_KEY가 없습니다.\n"
            ".env에 YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, ... 형식으로 추가하거나\n"
            "--api-key를 사용하세요."
        )
    return keys


def is_quota_exceeded(exc: HttpError) -> bool:
    reason = str(exc)
    return exc.resp.status == 403 and (
        "quotaExceeded" in reason or "dailyLimitExceeded" in reason
    )


def _print_summary(
    progress: ProgressTracker,
    prev_done_channels: int,
    total_channels: int,
    output_csv: Path,
    attempted_csv: Path,
) -> None:
    elapsed = time.monotonic() - progress.started_at
    this_run_channels = progress.completed_channels
    total_done = prev_done_channels + this_run_channels
    remaining = total_channels - total_done

    attempted_count = 0
    if attempted_csv.exists():
        try:
            attempted_count = sum(1 for _ in attempted_csv.open(encoding="utf-8-sig")) - 1
        except Exception:
            pass

    total_video_rows = 0
    if output_csv.exists():
        try:
            total_video_rows = sum(1 for _ in output_csv.open(encoding="utf-8-sig")) - 1
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("수집 현황 요약")
    print("=" * 60)
    print(f"  이번 실행 처리 채널 : {this_run_channels:>6}개")
    print(f"  이번 실행 저장 영상 : {progress.total_saved:>6}개")
    print(f"  누적 저장 영상 (행) : {total_video_rows:>6}개")
    print(f"  누적 완료 채널      : {total_done:>6}개 / {total_channels}개")
    print(f"  남은 채널           : {remaining:>6}개")
    print(f"  소요 시간           : {format_seconds(elapsed)}")
    print(f"  저장 파일           : {output_csv}")
    print(f"  attempted 파일      : {attempted_csv} ({attempted_count}개 기록)")
    print("=" * 60)


def _channel_worker(
    worker_id: int,
    api_key: str,
    channel_queue: queue.Queue,
    output_csv: Path,
    attempted_csv: Path,
    thumbnail_dir: Path,
    video_count: int,
    playlist_page_size: int,
    short_duration_sec: int,
    download_thumbnails: bool,
    thumbnail_workers: int,
    user_agent: str,
    shared_done_ids: set[str],
    done_ids_lock: threading.Lock,
    csv_lock: threading.Lock,
    print_lock: threading.Lock,
    progress: ProgressTracker,
    stop_event: threading.Event,
) -> None:
    youtube = build_youtube(api_key)

    while not stop_event.is_set():
        try:
            channel_name, channel_id = channel_queue.get_nowait()
        except queue.Empty:
            break

        with print_lock:
            print()
            print(progress.start_message(channel_name))
            print(f"  [키{worker_id}] 처리 중: {channel_name} ({channel_id})")

        with done_ids_lock:
            local_done_ids = set(shared_done_ids)

        try:
            channel_rows, stats = collect_channel_rows(
                youtube=youtube,
                channel_name=channel_name,
                channel_id=channel_id,
                video_count=video_count,
                playlist_page_size=playlist_page_size,
                done_ids=local_done_ids,
                short_duration_sec=short_duration_sec,
            )
        except HttpError as exc:
            if is_quota_exceeded(exc):
                with print_lock:
                    print(f"\n⚠ [키{worker_id}] 할당량 소진 → 이 스레드를 종료합니다.")
                    if all(
                        not t.is_alive()
                        for t in threading.enumerate()
                        if t is not threading.current_thread() and t.name.startswith("key-worker")
                    ):
                        stop_event.set()
                break
            with print_lock:
                print(f"  [키{worker_id}] HttpError ({channel_name}): {exc}")
            continue

        new_video_ids = {row["video_id"] for row in channel_rows}
        with done_ids_lock:
            shared_done_ids.update(new_video_ids)

        if download_thumbnails and channel_rows:
            download_thumbnails_batch(
                rows=channel_rows,
                thumbnail_dir=thumbnail_dir,
                user_agent=user_agent,
                max_workers=thumbnail_workers,
            )

        with csv_lock:
            append_rows(output_csv, channel_rows)
            append_attempted(attempted_csv, channel_id, stats["saved"])

        with print_lock:
            print(
                f"  [키{worker_id}] → 저장: {stats['saved']}개 | "
                f"스킵: Shorts {stats['skipped_shorts']} / "
                f"라이브 {stats['skipped_live']} / 기수집 {stats['skipped_done']}"
            )
            print(progress.finish_message(channel_name, stats["saved"]))


def resolve_channels_csv(path: Path) -> Path:
    if path.exists():
        return path

    alternatives = sorted(Path(".").glob("YT_ChannelData*_clean.csv"))
    alternatives.extend(sorted(Path(".").glob("YT_channelsList*.csv")))
    sample = ", ".join(str(p) for p in alternatives[:5])
    raise FileNotFoundError(
        f"채널 CSV를 찾을 수 없습니다: {path}\n"
        f"사용 가능한 후보 예시: {sample or '없음'}\n"
        "원하는 파일을 --channels-csv로 지정하세요."
    )


def build_youtube(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def sanitize_channel_name(channel_name: str) -> str:
    safe = "".join(c for c in channel_name if c.isalnum() or c in ("_", "-"))
    return safe or "unknown_channel"


def infer_uploads_playlist_id(channel_id: str) -> str | None:
    if channel_id.startswith("UC") and len(channel_id) > 2:
        return "UU" + channel_id[2:]
    return None


def get_uploads_playlist_id(youtube, channel_id: str) -> str | None:
    inferred = infer_uploads_playlist_id(channel_id)
    if inferred:
        return inferred

    response = youtube.channels().list(id=channel_id, part="contentDetails").execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_video_details_map(youtube, video_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not video_ids:
        return {}

    response = youtube.videos().list(
        id=",".join(video_ids),
        part="snippet,contentDetails,statistics,status,topicDetails",
        maxResults=min(len(video_ids), 50),
    ).execute()
    return {item["id"]: item for item in response.get("items", [])}


def get_thread_session(user_agent: str) -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        _thread_local.session = session
    return session


def compute_page_size(remaining_needed: int, playlist_page_size: int) -> int:
    # Fetch a modest over-sample so we do not classify or download far more videos than needed.
    target = max(remaining_needed * 10, 10)
    return max(1, min(playlist_page_size, target))

def download_thumbnail(
    thumb_url: str,
    local_path: Path,
    user_agent: str,
    timeout: int = 10,
    retries: int = 3,
) -> str:
    if not thumb_url:
        return ""

    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            resp = get_thread_session(user_agent).get(thumb_url, timeout=timeout)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return str(local_path)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (2**attempt))

    if last_error is not None:
        print(f"  → 썸네일 실패 ({local_path.name}): {last_error}")
    return ""


def download_thumbnails_batch(
    rows: list[dict[str, Any]],
    thumbnail_dir: Path,
    user_agent: str,
    max_workers: int,
) -> None:
    jobs: list[tuple[dict[str, Any], str, Path]] = []

    for row in rows:
        thumb_url = row["thumbnail_url"]
        if not thumb_url:
            row["thumbnail_path"] = ""
            continue

        safe_name = sanitize_channel_name(row["channel_name"])
        local_path = thumbnail_dir / safe_name / f"{row['video_id']}.jpg"
        if local_path.exists():
            row["thumbnail_path"] = str(local_path)
            continue

        jobs.append((row, thumb_url, local_path))

    if not jobs:
        return

    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(download_thumbnail, thumb_url, local_path, user_agent): row
            for row, thumb_url, local_path in jobs
        }
        for future, row in future_map.items():
            try:
                row["thumbnail_path"] = future.result()
            except Exception as exc:
                print(f"  → 썸네일 실패 ({row['video_id']}): {exc}")
                row["thumbnail_path"] = ""


def append_rows(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    write_header = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_done_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()

    existing = pd.read_csv(output_csv, usecols=["video_id"])
    return set(existing["video_id"].astype(str))


def load_done_channel_ids(output_csv: Path, video_count: int) -> set[str]:
    """video_count개 이상 수집이 완료된 channel_id 세트를 반환."""
    if not output_csv.exists():
        return set()

    existing = pd.read_csv(output_csv, usecols=["channel_id", "video_id"])
    counts = existing.groupby("channel_id")["video_id"].count()
    return set(counts[counts >= video_count].index.astype(str))


def load_attempted_channel_ids(attempted_csv: Path) -> set[str]:
    """이전에 시도했던 channel_id 세트를 반환."""
    if not attempted_csv.exists():
        return set()

    df = pd.read_csv(attempted_csv, usecols=["channel_id"])
    return set(df["channel_id"].astype(str))


def append_attempted(attempted_csv: Path, channel_id: str, saved_count: int) -> None:
    write_header = not attempted_csv.exists()
    with attempted_csv.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=["channel_id", "saved_count", "attempted_at"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "channel_id": channel_id,
            "saved_count": saved_count,
            "attempted_at": datetime.date.today().isoformat(),
        })


def is_live(snippet: dict[str, Any]) -> bool:
    return snippet.get("liveBroadcastContent") in ("live", "upcoming")


def parse_duration_seconds(duration: str | None) -> int | None:
    if not duration:
        return None

    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        duration,
    )
    if not match:
        return None

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def has_short_keyword(snippet: dict[str, Any]) -> bool:
    texts = [
        snippet.get("title") or "",
        snippet.get("description") or "",
        " ".join(snippet.get("tags", [])),
    ]
    return any(SHORT_KEYWORD_RE.search(text) for text in texts if text)


def is_short_video(item: dict[str, Any], short_duration_sec: int) -> bool:
    snippet = item.get("snippet", {})
    duration_sec = parse_duration_seconds(item.get("contentDetails", {}).get("duration"))
    if duration_sec is not None and duration_sec <= short_duration_sec:
        return True
    return has_short_keyword(snippet)


def build_row(channel_name: str, channel_id: str, item: dict[str, Any]) -> dict[str, Any]:
    snippet = item.get("snippet", {})
    content = item.get("contentDetails", {})
    stats = item.get("statistics", {})
    status = item.get("status", {})
    topics = item.get("topicDetails", {})
    thumbnails = snippet.get("thumbnails", {})
    thumb_info = thumbnails.get("maxres") or thumbnails.get("high") or {}

    return {
        "channel_name": channel_name,
        "channel_id": channel_id,
        "video_id": item["id"],
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "published_at": snippet.get("publishedAt"),
        "tags": ",".join(snippet.get("tags", [])),
        "category_id": snippet.get("categoryId"),
        "default_language": snippet.get("defaultLanguage", "N/A"),
        "duration": content.get("duration"),
        "dimension": content.get("dimension"),
        "definition": content.get("definition"),
        "caption": content.get("caption"),
        "view_count": stats.get("viewCount", 0),
        "like_count": stats.get("likeCount", 0),
        "favorite_count": stats.get("favoriteCount", 0),
        "comment_count": stats.get("commentCount", 0),
        "privacy_status": status.get("privacyStatus"),
        "license": status.get("license"),
        "embeddable": status.get("embeddable"),
        "made_for_kids": status.get("madeForKids"),
        "topic_categories": ",".join(topics.get("topicCategories", [])),
        "thumbnail_url": thumb_info.get("url", ""),
        "thumbnail_path": "",
    }


def collect_channel_rows(
    youtube,
    channel_name: str,
    channel_id: str,
    video_count: int,
    playlist_page_size: int,
    done_ids: set[str],
    short_duration_sec: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    uploads_id = get_uploads_playlist_id(youtube, channel_id)
    if not uploads_id:
        return [], {"saved": 0, "skipped_shorts": 0, "skipped_live": 0, "skipped_done": 0}

    rows: list[dict[str, Any]] = []
    skipped_shorts = 0
    skipped_live = 0
    skipped_done = 0
    next_page_token: str | None = None

    while len(rows) < video_count:
        remaining_needed = video_count - len(rows)
        request_kwargs: dict[str, Any] = {
            "playlistId": uploads_id,
            "part": "contentDetails",
            "maxResults": compute_page_size(remaining_needed, playlist_page_size),
        }
        if next_page_token:
            request_kwargs["pageToken"] = next_page_token

        try:
            playlist_resp = youtube.playlistItems().list(**request_kwargs).execute()
        except HttpError as exc:
            reason = getattr(exc, "reason", str(exc))
            if uploads_id != infer_uploads_playlist_id(channel_id):
                print(f"  → 재생목록 조회 실패 (스킵): {reason}")
                break

            try:
                fallback_resp = youtube.channels().list(id=channel_id, part="contentDetails").execute()
                items = fallback_resp.get("items", [])
                if not items:
                    print("  → 채널 없음, 스킵")
                    break
                uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
                request_kwargs["playlistId"] = uploads_id
                playlist_resp = youtube.playlistItems().list(**request_kwargs).execute()
            except HttpError as fallback_exc:
                print(f"  → 재생목록 조회 실패 (스킵): {getattr(fallback_exc, 'reason', str(fallback_exc))}")
                break

        page_ids = [
            item["contentDetails"]["videoId"]
            for item in playlist_resp.get("items", [])
            if item.get("contentDetails", {}).get("videoId")
        ]
        if not page_ids:
            break

        details_by_id = fetch_video_details_map(youtube, page_ids)
        for video_id in page_ids:
            if len(rows) >= video_count:
                break

            item = details_by_id.get(video_id)
            if item is None:
                continue
            if video_id in done_ids:
                skipped_done += 1
                continue
            if is_live(item.get("snippet", {})):
                skipped_live += 1
                continue
            if is_short_video(item, short_duration_sec):
                skipped_shorts += 1
                continue

            rows.append(build_row(channel_name, channel_id, item))
            done_ids.add(video_id)

        next_page_token = playlist_resp.get("nextPageToken")
        if not next_page_token:
            break

    stats = {
        "saved": len(rows),
        "skipped_shorts": skipped_shorts,
        "skipped_live": skipped_live,
        "skipped_done": skipped_done,
    }
    return rows, stats


def derive_category_name(channels_csv: Path) -> str:
    stem = channels_csv.stem  # e.g. "EDU_clean_1" or "VLOG"
    category = stem.split("_")[0].upper()
    return category


def youtube_data_extract(
    api_keys: list[str],
    channels_csv: Path,
    output_csv: Path,
    thumbnail_dir: Path,
    video_count: int,
    playlist_page_size: int = DEFAULT_PLAYLIST_PAGE_SIZE,
    thumbnail_workers: int = DEFAULT_THUMBNAIL_WORKERS,
    short_duration_sec: int = DEFAULT_SHORT_DURATION_SEC,
    download_thumbnails: bool = True,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    category = derive_category_name(channels_csv)
    thumbnail_dir = thumbnail_dir / f"{category}_thumbnails"

    channels_df = pd.read_csv(channels_csv)
    required_columns = {"channel_name", "channel_id"}
    missing = required_columns - set(channels_df.columns)
    if missing:
        raise ValueError(f"채널 CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    attempted_csv = output_csv.with_name(output_csv.stem + "_attempted.csv")

    done_ids = load_done_ids(output_csv)
    done_channel_ids = load_done_channel_ids(output_csv, video_count)
    attempted_channel_ids = load_attempted_channel_ids(attempted_csv)
    skip_channel_ids = done_channel_ids | attempted_channel_ids

    if skip_channel_ids:
        print(
            f"기존 데이터 로드: {len(done_ids)}개 영상 / "
            f"완료 채널 {len(done_channel_ids)}개 / "
            f"시도 완료 채널 {len(attempted_channel_ids)}개 (이어쓰기 모드)"
        )

    pending_df = channels_df[~channels_df["channel_id"].astype(str).isin(skip_channel_ids)]
    print(f"수집 대상: {len(pending_df)}개 채널 ({len(skip_channel_ids)}개 제외)")

    channel_queue: queue.Queue = queue.Queue()
    for row in pending_df.itertuples(index=False):
        channel_queue.put((str(row.channel_name), str(row.channel_id)))

    n_workers = len(api_keys)
    print(f"API 키 {n_workers}개로 병렬 처리 시작")

    shared_done_ids: set[str] = set(done_ids)
    done_ids_lock = threading.Lock()
    csv_lock = threading.Lock()
    print_lock = threading.Lock()
    stop_event = threading.Event()
    progress = ProgressTracker(total_channels=len(pending_df))

    threads = [
        threading.Thread(
            target=_channel_worker,
            name=f"key-worker-{i + 1}",
            args=(
                i + 1, api_keys[i], channel_queue,
                output_csv, attempted_csv, thumbnail_dir,
                video_count, playlist_page_size, short_duration_sec,
                download_thumbnails, thumbnail_workers, user_agent,
                shared_done_ids, done_ids_lock, csv_lock, print_lock, progress,
                stop_event,
            ),
            daemon=True,
        )
        for i in range(n_workers)
    ]

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n\n중단 신호 감지 → 현재 진행 중인 채널 완료 후 종료합니다...")
        stop_event.set()
        for t in threads:
            t.join()

    _print_summary(progress, len(skip_channel_ids), len(channels_df), output_csv, attempted_csv)

    if output_csv.exists():
        return pd.read_csv(output_csv, encoding="utf-8-sig")
    return pd.DataFrame(columns=FIELDNAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube 영상 메타데이터와 썸네일 수집")
    parser.add_argument("--api-key", help="YOUTUBE_API_KEY를 직접 전달할 때 사용")
    parser.add_argument(
        "--channels-csv",
        type=Path,
        default=DEFAULT_CHANNELS_CSV,
        help=f"채널 CSV 경로 (기본값: {DEFAULT_CHANNELS_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"결과 CSV 경로 (기본값: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=DEFAULT_THUMBNAIL_DIR,
        help=f"썸네일 저장 디렉터리 (기본값: {DEFAULT_THUMBNAIL_DIR})",
    )
    parser.add_argument(
        "--video-count",
        type=int,
        default=1,
        help="채널당 저장할 최대 일반 영상 수",
    )
    parser.add_argument(
        "--playlist-page-size",
        type=int,
        default=DEFAULT_PLAYLIST_PAGE_SIZE,
        help="playlistItems 한 번에 가져올 개수",
    )
    parser.add_argument(
        "--short-duration-sec",
        type=int,
        default=DEFAULT_SHORT_DURATION_SEC,
        help="이 길이 이하 영상은 Shorts로 간주합니다.",
    )
    parser.add_argument(
        "--thumbnail-workers",
        type=int,
        default=DEFAULT_THUMBNAIL_WORKERS,
        help="썸네일 다운로드 동시 요청 수",
    )
    parser.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="썸네일 다운로드를 건너뜁니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_keys = get_api_keys(args.api_key)
    channels_csv = resolve_channels_csv(args.channels_csv)

    youtube_data_extract(
        api_keys=api_keys,
        channels_csv=channels_csv,
        output_csv=args.output_csv,
        thumbnail_dir=args.thumbnail_dir,
        video_count=args.video_count,
        playlist_page_size=max(1, min(args.playlist_page_size, 50)),
        thumbnail_workers=max(1, args.thumbnail_workers),
        short_duration_sec=max(1, args.short_duration_sec),
        download_thumbnails=not args.skip_thumbnails,
    )


if __name__ == "__main__":
    main()
