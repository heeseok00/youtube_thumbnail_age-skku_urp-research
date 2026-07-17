# Conda environments (Linux, exported from server)

Recreated from the live server on 2026-06-27. Use these instead of the old root-level `ytvenv.yaml` (removed; it had Windows artifacts and outdated versions).

## Create environments

```bash
conda env create -f envs/ytvenv.yaml
conda env create -f envs/urp.yaml
conda env create -f envs/vlm_env.yaml
conda env create -f envs/urp_bin.yaml
```

Update an existing env after editing a yaml:

```bash
conda env update -n ytvenv -f envs/ytvenv.yaml --prune
```

Refresh exports from the server:

```bash
conda env export -n ytvenv --no-builds | grep -v '^prefix:' > envs/ytvenv.yaml
```

## Which env for what

| Env | Python | Main use |
|---|---|---|
| `ytvenv` | 3.10 | Thumbnail features (YOLO, EasyOCR, FER, MediaPipe), early clustering notebooks |
| `urp` | 3.9 | Grad-CAM, SHAP, XGBoost, deletion/insertion |
| `vlm_env` | 3.10 | Sentence-BERT, semantic/SHAP pipelines |
| `urp_bin` | 3.10 | CatBoost, PaddleOCR, highlight experiments |

## Not included in conda (install separately)

**Ollama** (thumbnail title OCR/correction in `3_썸네일 텍스트 추출 파이프라인/`):

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

**Hugging Face** (category classification in `step1_*.py`, `step2_classify_all.py`; LLaVA in `urp_yena/SOCIETY_caption.ipynb`):

- `Qwen/Qwen2.5-7B-Instruct`
- `llava-hf/llava-1.5-7b-hf`

Models download to `~/.cache/huggingface/` on first run. Set `HF_TOKEN` if rate-limited.

## Notes

- PyTorch CUDA builds in yaml may require the matching PyTorch index on install. If `conda env create` fails on torch lines, create the env without pip torch first, then install torch from https://pytorch.org for your CUDA version.
- These yamls were exported on **Linux x86_64**. They are not guaranteed to work on Windows/Mac without adjustment.
