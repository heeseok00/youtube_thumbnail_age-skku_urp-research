"""Step 1 — LLaVA-1.5-7b로 다이어트 표본 썸네일 캡션 생성.

예나(FOOD)의 캡셔닝 코드를 HEALTH 다이어트 표본에 맞게 튜닝한 버전.
- 입력: 00_sample/sampling.py의 load_diet_sample() (34-/65+ 각 1,250개, 시드 고정)
- 원본과 달리 자체 샘플링/파일명 딕셔너리 없이 표본의 resolved_path를 직접 사용
- 100개마다 자동 저장, 중단 후 재실행하면 이어서 진행 (ERROR 행 재시도 포함)
- 출력: outputs/diet_sample_2500_vlm.csv (vlm_caption 컬럼 추가)

실행: conda urp_yena 환경에서
  python 01_caption_diet.py
예상 소요: RTX 4090 기준 약 30분 (2,500장)
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent / "00_sample"))
from sampling import load_diet_sample  # noqa: E402

OUT_DIR = STAGE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUT_DIR / "diet_sample_2500_vlm.csv"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAVE_EVERY = 100

SYSTEM_PROMPT = (
    "USER: <image>\n"
    "Analyze the YouTube thumbnail carefully. "
    "If there are no people or clear subjects and the image consists only of text on a background, "
    "you MUST output exactly: 'Only text on the background.'\n"
    "Otherwise, describe the scene in 2-3 detailed sentences focusing strictly on the following: "
    "1. Who the main subject is and their exact facial expression or emotion. "
    "2. What action they are performing and what objects they are interacting with. "
    "3. The overall mood or atmosphere of the spatial background. "
    "Do not describe any text, typography, or color tones.\nASSISTANT:"
)


def main():
    # ── 이어하기: 기존 출력이 있으면 이어서, 없으면 공용 표본에서 시작 ──────
    if OUTPUT_CSV.exists():
        print(f"[이어하기] {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    else:
        df = load_diet_sample()
        df["target"] = df["age_group"].map({"34-": 0, "65+": 1})
        df["vlm_caption"] = None

    todo = df[
        df["vlm_caption"].isna()
        | df["vlm_caption"].astype(str).str.startswith("ERROR")
    ].index.tolist()
    print(f"캡셔닝 대상: {len(todo)} / {len(df)}")
    if not todo:
        print("모든 캡션 완료 상태입니다.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device} / 모델 로드 중: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    for i, idx in enumerate(tqdm(todo, desc="Captioning")):
        img_path = df.at[idx, "resolved_path"]
        try:
            raw_image = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(
                    text=SYSTEM_PROMPT, images=raw_image, return_tensors="pt"
                ).to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            full_text = processor.decode(output[0], skip_special_tokens=True)
            df.at[idx, "vlm_caption"] = full_text.split("ASSISTANT:")[-1].strip()
            del inputs, output
        except Exception as e:
            df.at[idx, "vlm_caption"] = f"ERROR: {e}"

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    n_err = df["vlm_caption"].astype(str).str.startswith("ERROR").sum()
    print(f"\n완료: {OUTPUT_CSV} (ERROR {n_err}건)")


if __name__ == "__main__":
    main()
