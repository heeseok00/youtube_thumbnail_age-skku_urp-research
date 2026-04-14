#!/usr/bin/env python3
"""Scrape viewer gender & age demographics from vling.net and append to clean CSV.

Usage:
    # 1) 최초 1회: 로그인 세션 저장
    python 04.vling_age_sex_extract.py --save-session

    # 2) 수집 실행 (카테고리 지정)
    python 04.vling_age_sex_extract.py --category VLOG
    python 04.vling_age_sex_extract.py --category SOCIETY

Input  : Data/{CATEGORY}/{CATEGORY}_clean.csv
Output : Data/{CATEGORY}/{CATEGORY}_age_sex.csv
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright
try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

CATEGORIES = ["VLOG", "FOOD", "EDU", "HEALTH", "SOCIETY"]
SESSION_FILE = Path("vling_session.json")
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 3.0


# ── vling 유틸 ────────────────────────────────────────────────

def to_vling_url(channel_id: str) -> str:
    return f"https://vling.net/channel/{channel_id}/viewers-info"


CAPTCHA_SELECTORS = [
    "iframe[src*='captcha']",
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    ".cf-challenge-running",
    "[class*='captcha']",
    "[id*='captcha']",
]


def check_and_handle_captcha(page) -> None:
    """캡차 감지 시 사용자가 직접 풀 때까지 대기."""
    for sel in CAPTCHA_SELECTORS:
        if page.query_selector(sel):
            print("\n  ⚠ 캡차 감지됨! 브라우저에서 직접 풀고 Enter를 누르세요...")
            input()
            print("  재개합니다.")
            return
    # 텍스트 기반 감지
    content = page.content()
    if "인간인지 확인" in content or "verify you are human" in content.lower():
        print("\n  ⚠ 캡차 감지됨! 브라우저에서 직접 풀고 Enter를 누르세요...")
        input()
        print("  재개합니다.")


def scrape_viewers_info(channel_id: str, channel_name: str, page) -> dict:
    result = {
        "female_pct": None, "male_pct": None,
        "age_~17": None, "age_18~24": None, "age_25~34": None,
        "age_35~44": None, "age_45~54": None, "age_55~64": None, "age_65~": None,
    }

    try:
        response = page.goto(to_vling_url(channel_id), wait_until="networkidle", timeout=30_000)

        # 캡차 체크
        check_and_handle_captcha(page)

        if response and response.status >= 400:
            print(f"  - 스킵 ({channel_name}): HTTP {response.status}")
            return result

        if page.query_selector('[class*="ViewersInfoWrapper_blurred"]'):
            print(f"  ⚠ 블러 ({channel_name}): 로그인 만료 — 세션 갱신 필요")
            return result

        try:
            page.wait_for_selector('[class*="GenderChart_genderTitle"]', timeout=10_000)
        except Exception:
            print(f"  - 데이터 없음 ({channel_name})")
            return result

        # 성별
        titles = page.query_selector_all('[class*="GenderChart_genderTitle"]')
        pcts   = page.query_selector_all('[class*="GenderChart_percent"]')
        print(f"    [DEBUG] 성별 타이틀 {len(titles)}개 / 퍼센트 {len(pcts)}개 발견")
        for title_el, pct_el in zip(titles, pcts):
            title = title_el.inner_text().strip()
            pct_text = pct_el.inner_text().replace("%", "").strip()
            print(f"    [DEBUG] 성별: '{title}' = '{pct_text}'")
            if not pct_text:
                continue
            pct = float(pct_text)
            if title == "여성":
                result["female_pct"] = pct
            elif title == "남성":
                result["male_pct"] = pct

        # 연령대
        age_rows = page.query_selector_all('[class*="AgeChart_age"]')
        print(f"    [DEBUG] 연령 행 {len(age_rows)}개 발견")
        for age_row in age_rows:
            label_el = age_row.query_selector('[class*="AgeChart_ageTitle"]')
            pct_el   = age_row.query_selector('[class*="AgeChart_percent"]')
            if label_el and pct_el:
                label = label_el.inner_text().strip()
                pct_text = pct_el.inner_text().replace("%", "").strip()
                print(f"    [DEBUG] 연령: '{label}' = '{pct_text}'")
                if label and pct_text:
                    result[f"age_{label}"] = float(pct_text)

    except Exception as e:
        print(f"  ✗ 오류 ({channel_name}): {e}")

    return result


# ── 세션 저장/복원 ────────────────────────────────────────────

def get_local_storage_js() -> str:
    return """() => {
        const o = {};
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            o[k] = localStorage.getItem(k);
        }
        return o;
    }"""


def save_session() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            locale="ko-KR", viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()
        page.goto("https://vling.net/", wait_until="networkidle", timeout=60_000)
        input("vling.net에 로그인 후 Enter를 누르세요...\n")

        SESSION_FILE.write_text(
            json.dumps(
                {"cookies": ctx.cookies(), "storage": page.evaluate(get_local_storage_js())},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"세션 저장 완료 → {SESSION_FILE}")
        browser.close()


def load_session(ctx, page) -> None:
    data    = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    cookies = data.get("cookies") or []
    storage = data.get("storage") or {}

    ctx.add_cookies(cookies)
    page.goto("https://vling.net/", wait_until="networkidle", timeout=60_000)

    if storage:
        page.evaluate(
            """(entries) => {
                for (const [k, v] of Object.entries(entries))
                    if (v != null) localStorage.setItem(k, String(v));
            }""",
            storage,
        )
    page.reload(wait_until="networkidle", timeout=60_000)


# ── 메인 수집 로직 ────────────────────────────────────────────

def _append_row(row: dict, out_path: Path) -> None:
    pd.DataFrame([row]).to_csv(
        out_path, mode="a", header=not out_path.exists(), index=False, encoding="utf-8-sig"
    )


def sex_age_extract(category: str) -> None:
    input_csv  = Path(f"Data/{category}/{category}_clean.csv")
    output_csv = Path(f"Data/{category}/{category}_age_sex.csv")

    if not input_csv.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_csv}")

    if not SESSION_FILE.exists():
        raise FileNotFoundError(
            f"세션 파일 없음: {SESSION_FILE}\n"
            "먼저 실행하세요: python 04.vling_age_sex_extract.py --save-session"
        )

    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    done_ids: set = set()
    if output_csv.exists():
        done_ids = set(pd.read_csv(output_csv, encoding="utf-8-sig")["channel_id"].dropna())

    targets = df[~df["channel_id"].isin(done_ids)].reset_index(drop=True)
    print(f"[{category}] 전체 {len(df)}개 | 완료 {len(done_ids)}개 | 수집 대상 {len(targets)}개")
    print(f"  입력: {input_csv}")
    print(f"  출력: {output_csv}\n")

    if targets.empty:
        print(f"[{category}] 모두 완료됨.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            locale="ko-KR", viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()
        if STEALTH_AVAILABLE:
            stealth_sync(page)
            print("  stealth 모드 적용됨")
        load_session(ctx, page)
        print("세션 복원 완료\n")

        completed = 0
        skipped = 0
        started_at = time.monotonic()

        def print_summary(reason: str) -> None:
            elapsed = int(time.monotonic() - started_at)
            total_done = len(done_ids) + completed
            remaining = len(df) - total_done
            m, s = divmod(elapsed, 60)
            print(f"\n{'─'*50}")
            print(f"[{category}] {reason}")
            print(f"  이번 실행 수집: {completed}개  (스킵/없음: {skipped}개)")
            print(f"  누적 완료:      {total_done}/{len(df)}개")
            print(f"  남은 채널:      {remaining}개")
            print(f"  소요 시간:      {m}분 {s}초")
            print(f"  출력 파일:      {output_csv}")
            print(f"{'─'*50}")

        try:
            for i, (_, row) in enumerate(targets.iterrows(), start=1):
                print(f"[{i}/{len(targets)}] {row['channel_name']}")
                viewers = scrape_viewers_info(row["channel_id"], row["channel_name"], page)
                _append_row({**row.to_dict(), **viewers}, output_csv)
                if all(v is None for v in viewers.values()):
                    skipped += 1
                else:
                    completed += 1
                time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
            print_summary("수집 완료")
        except KeyboardInterrupt:
            print_summary("중단됨 (Ctrl+C)")
        finally:
            try:
                browser.close()
            except Exception:
                pass


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vling.net 시청자 성별·연령 수집")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--save-session",
        action="store_true",
        help="vling.net 로그인 세션 저장 (최초 1회)",
    )
    group.add_argument(
        "--category",
        choices=CATEGORIES,
        help=f"수집할 카테고리: {CATEGORIES}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_session:
        save_session()
    else:
        sex_age_extract(args.category)


if __name__ == "__main__":
    main()
