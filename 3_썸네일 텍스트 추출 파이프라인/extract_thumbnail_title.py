"""
썸네일 이미지에서 제목 텍스트를 추출하여 CSV에 저장하는 파이프라인.

2단계 파이프라인:
  1단계: VLM(qwen2.5vl:7b)으로 썸네일에서 제목 텍스트 추출
  2단계: LLM(qwen2.5:7b)으로 영상 제목/설명 맥락을 참고해 오탈자 교정

입력:  Data/SOCIETY/SOCIETY_new_category_v3.csv
출력:  Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv
       (thumbnail_title 컬럼 추가)

체크포인트: Data/SOCIETY/ckpt_thumbnail_title.csv
           (중단 후 재실행 시 자동으로 이어서 진행)

사용법:
    cd /home/urp_jwl2/26-1_URP
    python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py"

    # 모델 변경 (기본: qwen2.5vl:7b)
    python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py" --vlm-model qwen2.5vl:32b

    # 교정 단계 생략 (빠른 실행)
    python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py" --no-correct
"""
# 파이프라인 참고: /home/urp_jwl2/urp_jungeun/thubnails_title_extract_ollama.ipynb

import os
import sys
import time
import base64
import argparse
import datetime
import requests
import pandas as pd
from tqdm import tqdm

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV   = os.path.join(BASE_DIR, "Data", "SOCIETY", "SOCIETY_new_category_v3.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR, "Data", "SOCIETY", "SOCIETY_new_category_v3_add_thumnail_title.csv")
CKPT_CSV    = os.path.join(BASE_DIR, "Data", "SOCIETY", "ckpt_thumbnail_title.csv")

# ── 기본 모델 설정 ───────────────────────────────────────────────────────────
DEFAULT_VLM_MODEL     = "qwen2.5vl:7b"     # 썸네일 → 텍스트 추출 (Vision 모델)
DEFAULT_CORRECT_MODEL = "qwen2.5:7b"       # 오탈자 교정 (텍스트 모델)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# ── 유틸 함수 ────────────────────────────────────────────────────────────────
def img_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_path(raw_path: str) -> str:
    """Windows 형식 경로 → Linux 절대 경로 변환"""
    normalized = raw_path.replace("\\", "/")
    return os.path.join(BASE_DIR, normalized)


OLLAMA_URL = DEFAULT_OLLAMA_URL  # main()에서 --ollama-url 인자로 덮어씀


def ollama_request(model: str, prompt: str, images: list = None,
                   timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048},
    }
    if images:
        payload["images"] = images

    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == 2:
                return f"ERROR: {e}"
            time.sleep(2 ** attempt)
    return "ERROR: max retries exceeded"


# ── 1단계: VLM으로 텍스트 추출 ──────────────────────────────────────────────
def extract_text_vlm(img_path: str, vlm_model: str, video_title: str = "") -> str:
    """썸네일 이미지에서 제목으로 추정되는 텍스트 추출"""
    if not os.path.exists(img_path):
        return "ERROR: 이미지 없음"

    hint = f"영상 제목 힌트: {video_title}\n" if video_title else ""
    prompt = (
        f"{hint}"
        "위 영상의 유튜브 썸네일입니다. "
        "이미지에서 실제로 보이는 텍스트만 추출해줘. "
        "힌트와 비슷한 텍스트가 보이면 참고하고, 보이지 않으면 빈 문자열을 출력해. "
        "특수기호도 전부 포함해서 추출하고, 없는 내용은 절대 만들어내지 마. "
        "설명 없이 추출한 텍스트만 출력해. /no_think"
    )
    try:
        b64 = img_to_base64(img_path)
    except Exception as e:
        return f"ERROR: {e}"

    return ollama_request(vlm_model, prompt, images=[b64])


# ── 2단계: LLM으로 오탈자 교정 ──────────────────────────────────────────────
def correct_text(raw_ocr: str, title: str, description: str,
                 correct_model: str) -> str:
    """영상 제목/설명 맥락을 참고해 OCR 오탈자 교정"""
    if not raw_ocr or raw_ocr.startswith("ERROR"):
        return raw_ocr

    desc_snippet = str(description)[:300] if pd.notna(description) else ""
    prompt = (
        f"영상 제목: {title}\n"
        f"영상 설명 일부: {desc_snippet}\n"
        f"OCR 추출 텍스트: {raw_ocr}\n\n"
        "위 맥락을 참고해서 OCR 텍스트의 오탈자/띄어쓰기만 수정해줘. "
        "내용은 바꾸지 말고, 수정된 텍스트만 출력해."
    )
    return ollama_request(correct_model, prompt)


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="썸네일 제목 텍스트 추출 파이프라인")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                        help=f"VLM 모델명 (기본: {DEFAULT_VLM_MODEL})")
    parser.add_argument("--correct-model", default=DEFAULT_CORRECT_MODEL,
                        help=f"교정 모델명 (기본: {DEFAULT_CORRECT_MODEL})")
    parser.add_argument("--no-correct", action="store_true",
                        help="2단계 오탈자 교정 생략")
    parser.add_argument("--batch-save", type=int, default=100,
                        help="체크포인트 저장 주기 (기본: 100건마다)")
    parser.add_argument("--subcategory", nargs="+", default=None,
                        help="처리할 subcategory 필터 (예: --subcategory 정치/선거/시위 불교/무속/운세)")
    parser.add_argument("--shard", type=int, default=0,
                        help="현재 샤드 번호 (0부터 시작, 기본: 0)")
    parser.add_argument("--total-shards", type=int, default=1,
                        help="전체 샤드 수 (기본: 1 = 샤딩 없음)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama API URL (기본: {DEFAULT_OLLAMA_URL})")
    args = parser.parse_args()

    # Ollama URL 전역 적용
    global OLLAMA_URL
    OLLAMA_URL = args.ollama_url.rstrip("/")
    if not OLLAMA_URL.endswith("/api/generate"):
        OLLAMA_URL = OLLAMA_URL + "/api/generate"

    # Ollama 서버 확인
    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("[오류] Ollama 서버가 실행 중이지 않습니다.")
        print("       'ollama serve' 를 먼저 실행하세요.")
        sys.exit(1)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print(f"[로드] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  전체 {len(df):,}행 로드 완료\n")

    # ── 샤드별 체크포인트 경로 설정 ─────────────────────────────────────────
    if args.total_shards > 1:
        shard_suffix = f"_shard{args.shard}of{args.total_shards}"
    else:
        shard_suffix = ""
    ckpt_path = CKPT_CSV.replace(".csv", f"{shard_suffix}.csv")

    # ── 체크포인트 로드 ──────────────────────────────────────────────────────
    if os.path.exists(ckpt_path):
        ckpt_df = pd.read_csv(ckpt_path, usecols=["video_id", "thumbnail_title"])
        done_map = dict(zip(ckpt_df["video_id"], ckpt_df["thumbnail_title"]))
        print(f"[체크포인트] {len(done_map):,}건 완료, 이어서 진행\n")
    else:
        done_map = {}
        print("[체크포인트] 없음, 처음부터 시작\n")

    # ── subcategory 필터 ─────────────────────────────────────────────────────
    if args.subcategory:
        mask = df["subcategory"].isin(args.subcategory)
        target_df = df[mask].copy()
        print(f"[필터] subcategory: {args.subcategory}")
        for cat in args.subcategory:
            cnt = (df["subcategory"] == cat).sum()
            print(f"  - {cat}: {cnt:,}건")
        print()
    else:
        target_df = df.copy()

    # ── 샤드 분할 ────────────────────────────────────────────────────────────
    if args.total_shards > 1:
        shard_indices = [i for i in range(len(target_df)) if i % args.total_shards == args.shard]
        target_df = target_df.iloc[shard_indices].copy()
        print(f"[샤드] {args.shard + 1}/{args.total_shards} — {len(target_df):,}건 담당\n")

    # ── 처리 대상 필터링 ─────────────────────────────────────────────────────
    todo_df = target_df[~target_df["video_id"].isin(done_map)].copy()

    print(f"  처리 현황:")
    print(f"    완료(체크포인트): {len(done_map):,}건")
    print(f"    남은 대상:        {len(todo_df):,}건")
    print(f"    총 대상:          {len(target_df):,}건\n")

    # ── OCR 추출 ─────────────────────────────────────────────────────────────
    if len(todo_df) > 0:
        print(f"[추출 시작] VLM={args.vlm_model}"
              + (f", 교정={args.correct_model}" if not args.no_correct else ", 교정=생략"))
        print(f"  대상: {len(todo_df):,}건")
        print(f"  시작: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        session_start = time.time()
        lap_times = []   # 최근 N건의 처리 시간 (이동 평균용)

        bar = tqdm(
            total=len(todo_df),
            desc="썸네일 OCR",
            unit="장",
            dynamic_ncols=True,
        )

        for i, (_, row) in enumerate(todo_df.iterrows()):
            t0 = time.time()
            img_path = resolve_path(row["thumbnail_path"])

            # 1단계: VLM 텍스트 추출 (영상 제목을 힌트로 전달)
            raw = extract_text_vlm(img_path, args.vlm_model,
                                   video_title=str(row.get("title", "")))

            # 2단계: 오탈자 교정
            if not args.no_correct and not raw.startswith("ERROR"):
                final = correct_text(raw, str(row.get("title", "")),
                                     row.get("description", ""),
                                     args.correct_model)
            else:
                final = raw

            done_map[row["video_id"]] = final

            # 시간 계산
            elapsed_item = time.time() - t0
            lap_times.append(elapsed_item)
            if len(lap_times) > 50:
                lap_times.pop(0)

            session_elapsed = time.time() - session_start
            avg_per_item    = sum(lap_times) / len(lap_times)
            remaining_items = len(todo_df) - (i + 1)
            eta_sec         = avg_per_item * remaining_items
            eta_str         = str(datetime.timedelta(seconds=int(eta_sec)))
            elapsed_str     = str(datetime.timedelta(seconds=int(session_elapsed)))
            total_done      = len(done_map)
            pct             = total_done / len(df) * 100

            bar.set_postfix_str(
                f"경과 {elapsed_str} | 예상 남은 시간 {eta_str} | "
                f"전체 {total_done:,}/{len(df):,} ({pct:.1f}%) | "
                f"{avg_per_item:.1f}s/장"
            )
            bar.update(1)

            # 체크포인트 저장
            if (i + 1) % args.batch_save == 0 or (i + 1) == len(todo_df):
                ckpt_rows = [{"video_id": vid, "thumbnail_title": txt}
                             for vid, txt in done_map.items()]
                pd.DataFrame(ckpt_rows).to_csv(ckpt_path, index=False)
                now = datetime.datetime.now().strftime("%H:%M:%S")
                tqdm.write(
                    f"  [{now}] 체크포인트 저장 — "
                    f"{total_done:,}/{len(df):,}건 완료 "
                    f"(경과 {elapsed_str}, 예상 남은 {eta_str})"
                )

        bar.close()
        total_session = str(datetime.timedelta(seconds=int(time.time() - session_start)))
        print(f"\n  이번 세션 소요 시간: {total_session}")

    # ── 최종 CSV 생성 ────────────────────────────────────────────────────────
    print("\n[완료] 최종 CSV 생성 중...")
    df["thumbnail_title"] = df["video_id"].map(done_map)

    null_cnt = df["thumbnail_title"].isna().sum()
    if null_cnt > 0:
        print(f"  [경고] thumbnail_title null {null_cnt:,}건 (추출 실패 또는 미처리)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  저장 완료 → {OUTPUT_CSV}")
    print(f"  총 {len(df):,}행 / {len(df.columns)}컬럼")

    # ── 결과 요약 ────────────────────────────────────────────────────────────
    filled = df["thumbnail_title"].notna().sum()
    empty  = (df["thumbnail_title"].astype(str).str.strip() == "").sum()
    error  = df["thumbnail_title"].astype(str).str.startswith("ERROR").sum()

    print(f"\n=== 결과 요약 ===")
    print(f"  텍스트 추출 성공: {filled - empty - error:,}건")
    print(f"  텍스트 없음(빈 썸네일): {empty:,}건")
    print(f"  오류: {error:,}건")
    print(f"  null: {null_cnt:,}건")


if __name__ == "__main__":
    main()
