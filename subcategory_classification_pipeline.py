#!/usr/bin/env python3
"""
SOCIETY 카테고리 소분류 분류 파이프라인 (통합본)

단계:
    1. discover   : 샘플 제목에서 소분류 후보 탐색 (Qwen2.5-7B)
    2. classify   : 전체 영상 1차 분류 (Qwen2.5-7B)
    3. refine     : GPT-4o-mini로 "기타" 행 재분류 + 카테고리 목록 사용자 확정
    4. reclassify : "기타" 행 강제 재분류 (GPT-4o-mini, 기타 선택지 없음)

사용법:
    python subcategory_classification_pipeline.py --category HEALTH --stage all
    python subcategory_classification_pipeline.py --category SOCIETY --stage discover
    python subcategory_classification_pipeline.py --category EDU     --stage classify

지원 카테고리: EDU, FOOD, HEALTH, SOCIETY, VLOG

결과물 (예: --category HEALTH):
    Data/HEALTH/category_candidates.json        (1단계 출력 — 2단계 자동 참조)
    Data/HEALTH/HEALTH_new_category.csv         (2단계 출력)
    Data/HEALTH/HEALTH_new_category_v2.csv      (3단계 출력)
    Data/HEALTH/HEALTH_new_category_v3.csv      (4단계 출력, 최종)

개선 사항 (기존 파일 대비):
    - step1/1b 결과가 step2에 자동 연결됨 (category_candidates.json 경유)
    - subcategory 컬럼명 단일화 (subcategory_refined 제거)
    - 실패 fallback이 인덱스 의존 아닌 find_closest() 사용
    - 체크포인트 간격 CKPT_INTERVAL로 통일
    - argparse로 단계별 실행 가능
"""

import argparse
import json
import os
import re
import sys
import time
from difflib import get_close_matches

import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── 고정 설정 ──────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPT_MODEL  = "gpt-4o-mini"

CKPT_INTERVAL   = 200   # 체크포인트 저장 간격 (행 단위, 모든 단계 통일)
DISCOVER_SAMPLE = 3000  # 후보 수집 시 최대 샘플 수

VALID_CATEGORIES = ["EDU", "FOOD", "HEALTH", "SOCIETY", "VLOG"]

# 카테고리별 한국어 표기 (프롬프트용)
CATEGORY_LABELS: dict[str, str] = {
    "EDU":     "EDU(교육)",
    "FOOD":    "FOOD(음식/요리)",
    "HEALTH":  "HEALTH(건강)",
    "SOCIETY": "SOCIETY(사회)",
    "VLOG":    "VLOG(일상/브이로그)",
}

# discover 단계를 건너뛸 때 사용하는 기본 카테고리 (카테고리별, 기타  제외)
FALLBACK_CATEGORIES: dict[str, list[str]] = {
    "SOCIETY": [
        "기독교/예배/찬양",
        "불교/무속/운세",
        "정치/선거/시위",
        "사회이슈/젠더/가족",
        "국제/해외반응",
        "시사/뉴스/사건",
        "건강/의료",
        "교육/입시",
        "스포츠/연예/문화",
        "역사/교양",
        "북한/안보/군사",
        "환경/과학/기술",
        "경제/부동산/재테크",
        "이슬람/다문화",
    ],
    "HEALTH": [],
    "EDU":    [],
    "FOOD":   [],
    "VLOG":   [],
}

# ── 동적 경로 (setup_paths() 호출 후 사용) ────────────────────────────────────

DATA_DIR        = ""
INPUT_CSV       = ""
CANDIDATES_JSON = ""
CLASSIFY_CSV    = ""
REFINE_CSV      = ""
FINAL_CSV       = ""
CKPT_DISCOVER   = ""
CKPT_CLASSIFY   = ""
CKPT_REFINE_S1  = ""
CKPT_REFINE_S3  = ""
CKPT_RECLS_S1   = ""
CKPT_RECLS_S3   = ""


def setup_paths(category: str) -> None:
    """카테고리에 따라 경로 전역 변수를 설정한다."""
    global DATA_DIR, INPUT_CSV, CANDIDATES_JSON, CLASSIFY_CSV, REFINE_CSV, FINAL_CSV
    global CKPT_DISCOVER, CKPT_CLASSIFY, CKPT_REFINE_S1, CKPT_REFINE_S3
    global CKPT_RECLS_S1, CKPT_RECLS_S3

    DATA_DIR        = os.path.join(BASE_DIR, "Data", category)
    INPUT_CSV       = os.path.join(DATA_DIR, f"{category}_final_kr_clean.csv")
    CANDIDATES_JSON = os.path.join(DATA_DIR, "category_candidates.json")
    CLASSIFY_CSV    = os.path.join(DATA_DIR, f"{category}_new_category.csv")
    REFINE_CSV      = os.path.join(DATA_DIR, f"{category}_new_category_v2.csv")
    FINAL_CSV       = os.path.join(DATA_DIR, f"{category}_new_category_v3.csv")

    CKPT_DISCOVER   = os.path.join(DATA_DIR, "ckpt_discover_chunks.json")
    CKPT_CLASSIFY   = os.path.join(DATA_DIR, "ckpt_classify.json")
    CKPT_REFINE_S1  = os.path.join(DATA_DIR, "ckpt_refine_s1.json")
    CKPT_REFINE_S3  = os.path.join(DATA_DIR, "ckpt_refine_s3.csv")
    CKPT_RECLS_S1   = os.path.join(DATA_DIR, "ckpt_recls_s1.json")
    CKPT_RECLS_S3   = os.path.join(DATA_DIR, "ckpt_recls_s3.csv")


# ── 공통 유틸 ──────────────────────────────────────────────────────────────────

def parse_json_safe(text: str):
    """GPT/Qwen 응답에서 JSON 추출. 마크다운 코드블록 포함 대응.
    여러 JSON 블록이 있을 경우 가장 긴 것을 선택 (짧은 [] 오파싱 방지).
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    candidates = re.findall(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
    if candidates:
        return json.loads(max(candidates, key=len))
    return json.loads(text)


def find_closest(cat: str, categories: list[str]) -> str:
    """카테고리명이 목록에 없을 때 가장 가까운 항목 반환."""
    if cat in categories:
        return cat
    matches = get_close_matches(cat, categories, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    matched = next((c for c in categories if cat in c or c in cat), None)
    if matched:
        return matched
    return categories[0]


def user_edit_categories(cats: list[str], existing: list[str] | None = None) -> list[str]:
    """CLI에서 카테고리 목록을 대화형으로 편집."""
    existing_set = set(existing or [])
    print()
    for i, c in enumerate(cats, 1):
        flag = "" if c in existing_set else "  ★ 신규"
        print(f"  {i:2}. {c}{flag}")
    print("\n  수정: 번호 새이름  |  추가: add 이름  |  삭제: del 번호  |  완료: ok\n")

    while True:
        cmd = input("  > ").strip()
        if cmd.lower() == "ok":
            break
        elif cmd.lower().startswith("add "):
            cat = cmd[4:].strip()
            cats.append(cat)
            print(f"    추가됨: {cat}")
        elif cmd.lower().startswith("del "):
            try:
                idx = int(cmd[4:].strip()) - 1
                removed = cats.pop(idx)
                print(f"    삭제됨: {removed}")
                for i, c in enumerate(cats, 1):
                    print(f"    {i:2}. {c}")
            except (ValueError, IndexError):
                print("    잘못된 입력입니다.")
        else:
            try:
                parts = cmd.split(" ", 1)
                idx = int(parts[0]) - 1
                cats[idx] = parts[1].strip()
                print(f"    수정됨: {cats[idx]}")
            except (ValueError, IndexError):
                print("    잘못된 입력입니다. (예: 3 새카테고리명)")

    return cats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_qwen() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Qwen2.5-7B 로드. VRAM이 부족하면 4-bit 양자화로 자동 전환."""
    print("Qwen2.5-7B 모델 로딩 중...")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB")
    else:
        vram_gb = 0
        print("  GPU 없음 — CPU로 실행 (매우 느림)")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)

    # Qwen2.5-7B bfloat16 ≈ 14 GB  →  VRAM 14 GB 미만이면 4-bit 양자화 사용
    if vram_gb > 0 and vram_gb < 14:
        print(f"  VRAM {vram_gb:.1f} GB < 14 GB → 4-bit 양자화 적용")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            device_map="auto" if DEVICE == "cuda" else "cpu",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    print("모델 로드 완료\n")
    return tokenizer, model


def qwen_chat(tokenizer, model, messages: list[dict], max_new_tokens: int = 500) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1단계: 카테고리 후보 탐색 (Qwen2.5-7B)
# ══════════════════════════════════════════════════════════════════════════════

def stage_discover(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    category: str = "SOCIETY",
) -> list[str]:
    """
    전체 데이터에서 2000개를 400개씩 5청크로 Qwen 분석 →
    종합하여 소분류 후보 12~15개 도출 → category_candidates.json 저장.
    이후 단계에서 해당 JSON을 자동으로 참조.
    """
    if os.path.exists(CANDIDATES_JSON):
        with open(CANDIDATES_JSON, encoding="utf-8") as f:
            cats = json.load(f)
        print(f"[1단계] 기존 후보 로드: {len(cats)}개 ({CANDIDATES_JSON})")
        return cats

    print("\n[1단계] 카테고리 후보 탐색 (Qwen2.5-7B) ...")

    SAMPLE_N = 1000
    CHUNK = 80   # VRAM 12GB 기준 안전 크기 (400 → 80)
    sample = df["title"].dropna().sample(
        n=min(SAMPLE_N, len(df)), random_state=42
    ).tolist()
    chunks = [sample[i:i + CHUNK] for i in range(0, len(sample), CHUNK)]

    # 청크 체크포인트 로드 (중간에 죽어도 이어서 실행)
    if os.path.exists(CKPT_DISCOVER):
        with open(CKPT_DISCOVER, encoding="utf-8") as f:
            chunk_results = json.load(f)
        print(f"  체크포인트: {len(chunk_results)}/{len(chunks)}청크 완료, 이어서 진행")
    else:
        chunk_results = []

    cat_label = CATEGORY_LABELS.get(category, category)
    for idx, chunk in enumerate(chunks[len(chunk_results):], len(chunk_results) + 1):
        numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(chunk)])
        messages = [
            {
                "role": "system",
                "content": "당신은 한국어 유튜브 콘텐츠를 분석하는 전문가입니다. 간결하게 답변해주세요.",
            },
            {
                "role": "user",
                "content": (
                    f"다음은 한국 유튜브 '{cat_label}' 카테고리 영상 제목 {len(chunk)}개입니다.\n\n"
                    f"{numbered}\n\n"
                    "위 제목들에서 등장하는 주제들을 간략히 나열해주세요.\n"
                    "형식: \"주제1, 주제2, 주제3, ...\" (쉼표로 구분)"
                ),
            },
        ]
        result = qwen_chat(tokenizer, model, messages, max_new_tokens=300)
        chunk_results.append(result)
        with open(CKPT_DISCOVER, "w", encoding="utf-8") as f:
            json.dump(chunk_results, f, ensure_ascii=False)
        print(f"  청크 {idx}/{len(chunks)} 완료")

    combined = "\n".join([f"청크{i+1}: {r}" for i, r in enumerate(chunk_results)])
    messages = [
        {
            "role": "system",
            "content": "당신은 한국어 유튜브 콘텐츠를 분석하는 전문가입니다.",
        },
        {
            "role": "user",
            "content": (
                f"다음은 {cat_label} 카테고리 영상 제목 {SAMPLE_N}개를 {CHUNK}개씩 분석한 결과입니다:\n\n"
                f"{combined}\n\n"
                "위 분석을 바탕으로 소분류를 12~15개로 정리해주세요. "
                "서로 겹치지 않고, 너무 세세하거나 광범위하지 않게 작성하세요.\n"
                "마지막에 반드시 JSON 배열로 출력하세요:\n"
                "```json\n[\"카테고리1\", \"카테고리2\", ...]\n```"
            ),
        },
    ]
    final_text = qwen_chat(tokenizer, model, messages, max_new_tokens=1000)

    try:
        cats = parse_json_safe(final_text)
        if not isinstance(cats, list):
            raise ValueError("리스트가 아님")
    except Exception:
        print("  JSON 파싱 실패 → 기본 카테고리 사용")
        cats = FALLBACK_CATEGORIES.copy()

    with open(CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(cats, f, ensure_ascii=False, indent=2)
    if os.path.exists(CKPT_DISCOVER):
        os.remove(CKPT_DISCOVER)
    print(f"  → {len(cats)}개 후보 저장 완료 ({CANDIDATES_JSON})\n")
    return cats


# ══════════════════════════════════════════════════════════════════════════════
# 2단계: 1차 전체 분류 (Qwen2.5-7B)
# ══════════════════════════════════════════════════════════════════════════════

def stage_classify(
    df: pd.DataFrame,
    categories: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> pd.DataFrame:
    """
    1단계에서 도출한 카테고리로 전체 영상을 Qwen이 20개씩 배치 분류.
    → subcategory 컬럼 생성 → CLASSIFY_CSV 저장.
    """
    if os.path.exists(CLASSIFY_CSV):
        result = pd.read_csv(CLASSIFY_CSV)
        print(f"[2단계] 기존 분류 결과 로드: {CLASSIFY_CSV}")
        return result

    print(f"\n[2단계] 1차 전체 분류 (Qwen2.5-7B) — {len(df):,}개 ...")

    if "기타" not in categories:
        categories = categories + ["기타"]
    cat_set = set(categories)
    cat_list_str = "\n".join(f"- {c}" for c in categories)

    def make_prompt(titles: list[str]) -> str:
        numbered = "\n".join([f'{i}: "{t}"' for i, t in enumerate(titles)])
        return (
            f"다음 유튜브 영상 제목들을 아래 카테고리 중 하나로 각각 분류하세요.\n\n"
            f"[카테고리]\n{cat_list_str}\n\n"
            f"[영상 제목]\n{numbered}\n\n"
            "규칙:\n"
            "- 반드시 위 카테고리 목록 중 정확히 하나만 선택\n"
            "- 판단하기 어려우면 \"기타\"\n"
            "- 반드시 아래 JSON 형식으로만 응답 (설명 없이)\n\n"
            f'형식: {{"0": "카테고리명", "1": "카테고리명", ...}}'
        )

    def parse_batch(text: str, n: int) -> list[str]:
        try:
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                result = []
                for i in range(n):
                    cat = obj.get(str(i), "기타").strip()
                    if cat not in cat_set:
                        cat = find_closest(cat, categories)
                    result.append(cat)
                return result
        except Exception:
            pass
        return ["기타"] * n

    titles = df["title"].fillna("").tolist()
    total = len(titles)
    BATCH = 10   # VRAM 12GB 기준 안전 크기 (20 → 10)
    batches = [titles[i:i + BATCH] for i in range(0, total, BATCH)]

    # 체크포인트 로드
    results: dict[int, str] = {}
    start_batch = 0
    if os.path.exists(CKPT_CLASSIFY):
        with open(CKPT_CLASSIFY, encoding="utf-8") as f:
            ckpt = json.load(f)
        results = {int(k): v for k, v in ckpt.get("results", {}).items()}
        start_batch = ckpt.get("next_batch", 0)
        print(f"  체크포인트: {len(results):,}개 처리됨, 배치 {start_batch}부터 재개\n")

    t0 = time.time()
    error_count = 0

    for batch_idx in range(start_batch, len(batches)):
        batch_titles = batches[batch_idx]
        global_start = batch_idx * BATCH

        try:
            messages = [
                {
                    "role": "system",
                    "content": "당신은 한국어 유튜브 영상 제목을 보고 주제를 분류하는 전문가입니다. 반드시 주어진 카테고리 중 하나만 선택하세요.",
                },
                {"role": "user", "content": make_prompt(batch_titles)},
            ]
            response = qwen_chat(tokenizer, model, messages, max_new_tokens=300)
            cats = parse_batch(response, len(batch_titles))
        except Exception as e:
            print(f"  [오류] 배치 {batch_idx}: {e}")
            cats = ["기타"] * len(batch_titles)
            error_count += 1

        for local_i, cat in enumerate(cats):
            results[global_start + local_i] = cat

        # 진행률 출력
        done = len(results)
        elapsed = time.time() - t0
        batches_done = batch_idx - start_batch + 1
        speed = batches_done / elapsed * BATCH if elapsed > 0 else 0
        remaining = (total - done) / speed if speed > 0 else 0
        pct = done / total * 100
        bar = "█" * int(30 * pct / 100) + "░" * (30 - int(30 * pct / 100))
        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}% | {done:,}/{total:,}개 | "
            f"{speed:.0f}개/s | 남은 {remaining/60:.1f}분 | 오류 {error_count}건"
        )
        sys.stdout.flush()

        # 체크포인트 저장 (배치 인덱스 기준으로 안정적으로 판단)
        if (batch_idx + 1) % (CKPT_INTERVAL // BATCH) == 0:
            with open(CKPT_CLASSIFY, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": results, "next_batch": batch_idx + 1},
                    f, ensure_ascii=False,
                )

    print()
    df = df.copy()
    df["subcategory"] = [results.get(i, "기타") for i in range(total)]
    df.to_csv(CLASSIFY_CSV, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {CLASSIFY_CSV}\n")

    if os.path.exists(CKPT_CLASSIFY):
        os.remove(CKPT_CLASSIFY)

    print("=== 소분류별 영상 수 ===")
    for cat, cnt in df["subcategory"].value_counts().items():
        print(f"  {cat}: {cnt:,}개 ({cnt/total*100:.1f}%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GPT 공통 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _gpt_discover_candidates(
    client: OpenAI, df: pd.DataFrame, batch_size: int = 500
) -> list[str]:
    """df를 batch_size 단위로 GPT에게 보내 카테고리 후보 수집."""
    system = (
        "당신은 한국 유튜브 영상 분류 전문가입니다.\n"
        "아래 영상 목록을 보고 어떤 카테고리들이 필요한지 JSON 배열로 제안하세요.\n"
        "형식: [\"카테고리1\", \"카테고리2\", ...]\n"
        "규칙: 10~20개 사이로, 한국어로, 구체적으로 작성하세요."
    )
    candidates: list[str] = []
    total = len(df)
    t0 = time.time()
    n_batches = (total + batch_size - 1) // batch_size

    has_tags = "tags" in df.columns
    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        chunk = df.iloc[start:start + batch_size]
        lines = []
        for row in chunk.itertuples():
            tags = str(getattr(row, "tags", ""))[:100] if has_tags and pd.notna(getattr(row, "tags", None)) else ""
            lines.append(f"- 제목: {str(row.title)[:80]} | 태그: {tags}")
        user_msg = "\n".join(lines)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3, max_tokens=500,
                )
                result = parse_json_safe(resp.choices[0].message.content)
                if isinstance(result, list):
                    candidates.extend(result)
                    break
            except Exception as e:
                print(f"    재시도 {attempt+1}/3: {e}")
                time.sleep(2 ** attempt)

        done = min(start + batch_size, total)
        elapsed = time.time() - t0
        speed = batch_num / elapsed if elapsed > 0 else 0
        remaining = (n_batches - batch_num) / speed if speed > 0 else 0
        elapsed_str = f"{int(elapsed//60)}분 {int(elapsed%60)}초"
        remain_str  = f"{int(remaining//60)}분 {int(remaining%60)}초"
        pct = done / total * 100
        bar = "█" * int(20 * pct / 100) + "░" * (20 - int(20 * pct / 100))
        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}% | {done:,}/{total:,}개 | "
            f"경과 {elapsed_str} | 남은 {remain_str} | 후보 {len(candidates)}개"
        )
        sys.stdout.flush()
        time.sleep(0.3)

    print()
    return candidates


def _gpt_consolidate(
    client: OpenAI,
    candidates: list,
    existing: list[str],
    allow_gita: bool,
) -> list[str]:
    """
    수집된 후보를 GPT가 정리 → 사용자가 CLI에서 최종 확정.
    allow_gita=False 이면 '기타' 항목 제거.
    """
    flat = []
    for item in candidates:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, str):
            flat.append(item)
    unique = list(dict.fromkeys(flat))

    system = (
        "당신은 한국 유튜브 채널 분류 전문가입니다.\n"
        f"기존 카테고리(반드시 그대로 유지): {json.dumps(existing, ensure_ascii=False)}\n\n"
        "아래 후보를 분석해서 기존으로 흡수 불가능한 완전히 새로운 유형만 추가하세요.\n"
        "예) '정치 토론', '정치 비판'은 모두 '정치/선거/시위'로 흡수됩니다.\n"
        "규칙:\n"
        "- 기존 카테고리 이름을 한 글자도 바꾸지 말고 그대로 포함\n"
        "- 기존으로 커버 안 되는 신규 유형만 추가 (최대 6개)\n"
        + ("- '기타'를 마지막에 포함하세요.\n" if allow_gita
           else "- '기타'는 절대 포함하지 마세요.\n")
        + "- JSON 배열로만 반환하세요."
    )

    final_cats = unique[:25]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(unique, ensure_ascii=False)},
                ],
                temperature=0, max_tokens=800,
            )
            result = parse_json_safe(resp.choices[0].message.content)
            if isinstance(result, list):
                final_cats = result
                break
        except Exception as e:
            print(f"  재시도 {attempt+1}/3: {e}")
            time.sleep(2 ** attempt)

    if not allow_gita:
        final_cats = [c for c in final_cats if c != "기타"]

    print(f"\n  GPT 제안 카테고리 {len(final_cats)}개 (★ = 신규):")
    final_cats = user_edit_categories(final_cats, existing)

    if not allow_gita:
        final_cats = [c for c in final_cats if c != "기타"]

    print(f"\n  확정 카테고리 {len(final_cats)}개:")
    for i, c in enumerate(final_cats, 1):
        flag = "" if c in set(existing) else "  ★ 신규"
        print(f"    {i:2}. {c}{flag}")
    return final_cats


def _gpt_classify_rows(
    client: OpenAI,
    df: pd.DataFrame,
    categories: list[str],
    ckpt_path: str,
    allow_gita: bool,
) -> pd.DataFrame:
    """
    df의 모든 행을 GPT로 분류 → subcategory 컬럼으로 반환.
    ckpt_path에 CKPT_INTERVAL 행마다 저장.
    """
    BATCH = 10
    cat_set = set(categories)

    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path, index_col=0)
        done_idx = set(ckpt.index.tolist())
        print(f"  체크포인트: {len(done_idx):,}개 완료, 이어서 진행")
    else:
        ckpt = pd.DataFrame(columns=["subcategory"])
        done_idx = set()

    has_tags = "tags" in df.columns
    todo = df[~df.index.isin(done_idx)]
    indices = todo.index.tolist()
    print(f"  남은 행: {len(indices):,}개\n")

    system = (
        "당신은 한국 유튜브 영상 분류 전문가입니다.\n"
        "영상의 제목과 태그를 보고 아래 카테고리 중 가장 적합한 것을 선택하세요.\n\n"
        "카테고리:\n"
        + "\n".join(f"- {c}" for c in categories)
        + "\n\n규칙:\n"
        "1. 위 목록 중 하나만 선택하세요.\n"
        + (
            "2. '기타'는 정말 어디에도 해당하지 않을 때만 사용하세요.\n"
            if allow_gita else
            "2. '기타'는 없습니다. 애매해도 가장 가까운 것을 선택하세요.\n"
        )
        + "3. JSON 배열로만 응답하세요. 예: [\"카테고리1\", \"카테고리2\"]\n"
        "4. 입력 순서와 동일하게 반환하세요."
    )

    results = []
    n_done = 0
    t0 = time.time()
    total_todo = len(indices)

    for start in range(0, total_todo, BATCH):
        batch_idx = indices[start:start + BATCH]
        cols = ["title"] + (["tags"] if has_tags else [])
        batch_rows = todo.loc[batch_idx, cols].to_dict("records")
        lines = []
        for i, row in enumerate(batch_rows, 1):
            tags = str(row.get("tags", "")) if pd.notna(row.get("tags", "")) else ""
            lines.append(
                f"{i}. 제목: {str(row.get('title', ''))[:100]}\n   태그: {tags[:200]}"
            )
        user_msg = "\n\n".join(lines)

        success = False
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0, max_tokens=400,
                )
                cats = parse_json_safe(resp.choices[0].message.content)
                # 중첩 리스트 평탄화
                if isinstance(cats, list):
                    cats = [
                        c[0] if isinstance(c, list) else str(c) for c in cats
                    ]
                if isinstance(cats, list) and len(cats) == len(batch_idx):
                    for idx, cat in zip(batch_idx, cats):
                        cat = cat.strip()
                        assigned = cat if cat in cat_set else find_closest(cat, categories)
                        results.append({"index": idx, "subcategory": assigned})
                    success = True
                    break
            except Exception as e:
                print(f"    재시도 {attempt+1}/3: {e}")
                time.sleep(2 ** attempt)

        if not success:
            fallback = find_closest("시사/뉴스/사건", categories)
            for idx in batch_idx:
                results.append({"index": idx, "subcategory": fallback})

        n_done += len(batch_idx)

        # 진행률 출력
        elapsed = time.time() - t0
        speed = n_done / elapsed if elapsed > 0 else 0
        remaining = (total_todo - n_done) / speed if speed > 0 else 0
        pct = n_done / total_todo * 100
        bar = "█" * int(30 * pct / 100) + "░" * (30 - int(30 * pct / 100))
        elapsed_str = f"{int(elapsed//60)}분 {int(elapsed%60)}초"
        remain_str  = f"{int(remaining//60)}분 {int(remaining%60)}초"
        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}% | {n_done:,}/{total_todo:,}개 | "
            f"경과 {elapsed_str} | 남은 {remain_str}"
        )
        sys.stdout.flush()

        # 마지막 배치는 항상 저장, 그 외는 CKPT_INTERVAL 배수일 때만 저장 (이중 저장 방지)
        is_last = n_done == total_todo
        if is_last or (n_done % CKPT_INTERVAL == 0):
            new_df = pd.DataFrame(results).set_index("index")
            ckpt = pd.concat([ckpt, new_df]).drop_duplicates()
            ckpt.to_csv(ckpt_path)
            results = []

        time.sleep(0.4)

    print()

    return pd.read_csv(ckpt_path, index_col=0)


# ══════════════════════════════════════════════════════════════════════════════
# 3단계: GPT 전체 재분류
# ══════════════════════════════════════════════════════════════════════════════

def stage_refine(
    df: pd.DataFrame, base_categories: list[str], client: OpenAI
) -> pd.DataFrame:
    """
    2단계에서 "기타"로 분류된 행만 GPT-4o-mini로 재분류.
      - "기타" 행에서 카테고리 후보 수집
      - 사용자가 CLI에서 최종 목록 확정 (기타 허용)
      - 확정 카테고리로 "기타" 행만 재분류 후 REFINE_CSV 저장
    """
    if os.path.exists(REFINE_CSV):
        result = pd.read_csv(REFINE_CSV)
        print(f"[3단계] 기존 재분류 결과 로드: {REFINE_CSV}")
        return result

    gita_df = df[df["subcategory"] == "기타"].copy()
    print(f"\n[3단계] GPT '기타' 재분류 — {len(gita_df):,}개 ...")

    if len(gita_df) == 0:
        print("  '기타' 행 없음, 건너뜀")
        df.to_csv(REFINE_CSV, index=False, encoding="utf-8-sig")
        return df

    # 후보 수집 ("기타" 행에서 최대 DISCOVER_SAMPLE개 샘플링)
    if os.path.exists(CKPT_REFINE_S1):
        with open(CKPT_REFINE_S1, encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"  [체크포인트] 후보 {len(candidates)}개 로드")
    else:
        sample_df = (
            gita_df.sample(n=DISCOVER_SAMPLE, random_state=42)
            if len(gita_df) > DISCOVER_SAMPLE else gita_df
        )
        print(f"  [Step 3-1] '기타' 행에서 카테고리 후보 수집 중 ({len(sample_df):,}개 샘플)...")
        candidates = _gpt_discover_candidates(client, sample_df, batch_size=300)
        with open(CKPT_REFINE_S1, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)

    # 2단계에서 실제 사용된 카테고리를 existing으로 사용 (기타 제외)
    existing_cats = [c for c in df["subcategory"].unique() if c != "기타"]

    # 통합 + 사용자 확정 (기타 허용)
    print("\n  [Step 3-2] 카테고리 목록 확정 중...")
    final_cats = _gpt_consolidate(client, candidates, existing_cats, allow_gita=True)

    # "기타" 행만 재분류
    print("\n  [Step 3-3] '기타' 행 재분류 중...")
    result_df = _gpt_classify_rows(client, gita_df, final_cats, CKPT_REFINE_S3, allow_gita=True)

    df = df.copy()
    df["subcategory"] = result_df["subcategory"].combine_first(df["subcategory"])

    df.to_csv(REFINE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  저장 완료: {REFINE_CSV}\n")

    # 체크포인트 정리
    for ckpt in [CKPT_REFINE_S1, CKPT_REFINE_S3]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    print("=== subcategory 분포 ===")
    for cat, cnt in df["subcategory"].value_counts().items():
        print(f"  {cat:<30} {cnt:>6}개  ({cnt/len(df)*100:.1f}%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4단계: "기타" 강제 재분류
# ══════════════════════════════════════════════════════════════════════════════

def stage_reclassify(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    subcategory == "기타"인 행만 대상으로,
    "기타" 선택지 없이 GPT-4o-mini가 강제 재분류.
    → FINAL_CSV 저장 (최종 결과물).
    """
    if os.path.exists(FINAL_CSV):
        result = pd.read_csv(FINAL_CSV)
        print(f"[4단계] 기존 최종 결과 로드: {FINAL_CSV}")
        return result

    gita_df = df[df["subcategory"] == "기타"].copy()
    print(f"\n[4단계] '기타' 강제 재분류 — {len(gita_df):,}개 ...")

    if len(gita_df) == 0:
        print("  '기타' 행 없음, 건너뜀")
        df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        return df

    existing_cats = [c for c in df["subcategory"].unique() if c != "기타"]

    # 후보 수집 ("기타" 행에서 최대 DISCOVER_SAMPLE개 샘플링)
    if os.path.exists(CKPT_RECLS_S1):
        with open(CKPT_RECLS_S1, encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"  [체크포인트] 후보 {len(candidates)}개 로드")
    else:
        sample_df = (
            gita_df.sample(n=DISCOVER_SAMPLE, random_state=42)
            if len(gita_df) > DISCOVER_SAMPLE else gita_df
        )
        print(f"  [Step 4-1] '기타' 행에서 카테고리 후보 수집 중 ({len(sample_df):,}개 샘플)...")
        candidates = _gpt_discover_candidates(client, sample_df, batch_size=300)
        with open(CKPT_RECLS_S1, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)

    # 통합 + 사용자 확정 (기타 없음)
    print("\n  [Step 4-2] 카테고리 목록 확정 중 ('기타' 없음)...")
    final_cats = _gpt_consolidate(
        client, candidates, existing_cats, allow_gita=False
    )

    # 강제 분류
    print("\n  [Step 4-3] '기타' 행 강제 분류 중...")
    result_df = _gpt_classify_rows(
        client, gita_df, final_cats, CKPT_RECLS_S3, allow_gita=False
    )

    df = df.copy()
    df["subcategory"] = result_df["subcategory"].combine_first(df["subcategory"])

    # 잔재 이상 카테고리 정리
    all_valid = set(existing_cats) | set(final_cats)
    invalid_mask = ~df["subcategory"].isin(all_valid)
    if invalid_mask.sum() > 0:
        fallback = find_closest("시사/뉴스/사건", list(all_valid))
        print(f"\n  잔재 이상 카테고리 {invalid_mask.sum()}개 → '{fallback}'으로 처리")
        df.loc[invalid_mask, "subcategory"] = fallback

    df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  저장 완료: {FINAL_CSV}\n")

    # 체크포인트 정리
    for ckpt in [CKPT_RECLS_S1, CKPT_RECLS_S3]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    print("=== 최종 subcategory 분포 ===")
    for cat, cnt in df["subcategory"].value_counts().items():
        flag = "  ★ 신규" if cat not in set(existing_cats) else ""
        print(f"  {cat:<30} {cnt:>6}개  ({cnt/len(df)*100:.1f}%){flag}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 인구통계 요약 리포트
# ══════════════════════════════════════════════════════════════════════════════

def report_demographics(df: pd.DataFrame) -> None:
    """소분류별 연령(~34 vs 65~) 비율과 성별 비율을 출력한다."""
    if "subcategory" not in df.columns:
        print("[리포트] subcategory 컬럼 없음 — 건너뜀")
        return

    AGE_COLS = ["age_~17", "age_18~24", "age_25~34",
                "age_35~44", "age_45~54", "age_55~64", "age_65~"]
    missing_age = [c for c in AGE_COLS if c not in df.columns]
    missing_sex = [c for c in ["female_pct", "male_pct"] if c not in df.columns]
    if missing_age or missing_sex:
        print(f"[리포트] 연령/성별 컬럼 없음: {missing_age + missing_sex}")
        return

    # 연령 5그룹 합산
    df = df.copy()
    df["age_~34"] = df["age_~17"] + df["age_18~24"] + df["age_25~34"]
    df["age_65~"] = df["age_65~"]

    # 연령 데이터가 있는 행만 사용
    has_age = df[AGE_COLS].notna().all(axis=1)
    d = df[has_age].copy()

    print("\n" + "=" * 70)
    print("  소분류별 인구통계 요약")
    print("=" * 70)
    print(f"  {'소분류':<25} {'영상수':>6}  {'~34%':>6}  {'65~%':>6}  {'여성%':>6}  {'남성%':>6}")
    print("-" * 70)

    for cat, grp in d.groupby("subcategory"):
        n         = len(grp)
        young_pct = grp["age_~34"].mean()
        old_pct   = grp["age_65~"].mean()
        f_pct     = grp["female_pct"].mean() if grp["female_pct"].notna().any() else float("nan")
        m_pct     = grp["male_pct"].mean()   if grp["male_pct"].notna().any()   else float("nan")
        print(
            f"  {str(cat):<25} {n:>6,}  "
            f"{young_pct:>5.1f}%  {old_pct:>5.1f}%  "
            f"{f_pct:>5.1f}%  {m_pct:>5.1f}%"
        )

    print("-" * 70)
    print(f"  {'전체 평균':<25} {len(d):>6,}  "
          f"{d['age_~34'].mean():>5.1f}%  {d['age_65~'].mean():>5.1f}%  "
          f"{d['female_pct'].mean():>5.1f}%  {d['male_pct'].mean():>5.1f}%")
    print("=" * 70)
    print(f"  ※ 연령 데이터 없는 영상 {(~has_age).sum():,}개 제외\n")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="유튜브 카테고리 소분류 분류 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "단계 설명:\n"
            "  discover   : Qwen으로 카테고리 후보 탐색 → category_candidates.json\n"
            "  classify   : Qwen으로 전체 1차 분류    → {CATEGORY}_new_category.csv\n"
            "  refine     : GPT로 '기타' 재분류        → {CATEGORY}_new_category_v2.csv\n"
            "  reclassify : GPT로 '기타' 강제 재분류   → {CATEGORY}_new_category_v3.csv\n"
            "  all        : 위 4단계 순차 실행 (기본값)\n\n"
            "예시:\n"
            "  python subcategory_classification_pipeline.py --category HEALTH --stage all\n"
        ),
    )
    parser.add_argument(
        "--category",
        choices=VALID_CATEGORIES,
        default="SOCIETY",
        help="처리할 카테고리 (기본값: SOCIETY)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="데이터 루트 디렉터리 (기본값: 스크립트 위치)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "discover", "classify", "refine", "reclassify"],
        default="all",
    )
    args = parser.parse_args()

    # 카테고리에 따라 경로 설정
    global BASE_DIR
    if args.base_dir:
        BASE_DIR = os.path.abspath(args.base_dir)
    setup_paths(args.category)

    run_all        = args.stage == "all"
    run_discover   = run_all or args.stage == "discover"
    run_classify   = run_all or args.stage == "classify"
    run_refine     = run_all or args.stage == "refine"
    run_reclassify = run_all or args.stage == "reclassify"

    print(f"카테고리: {args.category} ({CATEGORY_LABELS.get(args.category, args.category)})")
    print(f"데이터 경로: {DATA_DIR}\n")
    print("CSV 로딩 중...")
    df_raw = pd.read_csv(INPUT_CSV)
    print(f"총 {len(df_raw):,}개 행 로드 완료\n")

    # GPT 클라이언트는 필요한 단계에서만 생성
    client = None
    if run_refine or run_reclassify:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = input("OpenAI API 키 입력 (sk-...): ").strip()
        client = OpenAI(api_key=api_key)

    # Qwen 모델은 1·2단계가 모두 필요할 때 한 번만 로드
    qwen_tokenizer, qwen_model = None, None
    if run_discover or run_classify:
        qwen_tokenizer, qwen_model = load_qwen()

    # 1단계: 카테고리 후보 탐색
    categories = FALLBACK_CATEGORIES.get(args.category, []).copy()
    if run_discover:
        categories = stage_discover(df_raw, qwen_tokenizer, qwen_model, args.category)
    elif os.path.exists(CANDIDATES_JSON):
        with open(CANDIDATES_JSON, encoding="utf-8") as f:
            categories = json.load(f)
        print(f"category_candidates.json 로드: {len(categories)}개\n")

    if not categories:
        print(
            f"[경고] {args.category} 카테고리의 기본 카테고리 목록이 없습니다.\n"
            "  --stage discover 를 먼저 실행하거나 category_candidates.json을 준비하세요."
        )

    # 2단계: 1차 분류
    df = df_raw
    if run_classify:
        df = stage_classify(df_raw, categories, qwen_tokenizer, qwen_model)
    elif os.path.exists(CLASSIFY_CSV):
        df = pd.read_csv(CLASSIFY_CSV)
        print(f"{os.path.basename(CLASSIFY_CSV)} 로드: {len(df):,}개\n")

    # 3단계: GPT 재분류
    if run_refine:
        df = stage_refine(df, categories, client)
    elif os.path.exists(REFINE_CSV):
        df = pd.read_csv(REFINE_CSV)
        print(f"{os.path.basename(REFINE_CSV)} 로드: {len(df):,}개\n")

    # 4단계: 기타 강제 재분류
    if run_reclassify:
        df = stage_reclassify(df, client)

    print("\n파이프라인 완료.")
    print(f"최종 결과: {FINAL_CSV}")

    # 최종 결과가 있으면 인구통계 요약 출력
    result_path = FINAL_CSV if os.path.exists(FINAL_CSV) else (
        REFINE_CSV if os.path.exists(REFINE_CSV) else (
        CLASSIFY_CSV if os.path.exists(CLASSIFY_CSV) else None)
    )
    if result_path:
        report_demographics(pd.read_csv(result_path))


if __name__ == "__main__":
    main()
