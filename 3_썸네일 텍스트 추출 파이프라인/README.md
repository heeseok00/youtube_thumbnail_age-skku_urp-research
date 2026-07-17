# 3. 썸네일 텍스트 추출 파이프라인

유튜브 썸네일 이미지에서 제목 텍스트를 추출하여 `thumbnail_title` 컬럼으로 저장합니다.

## 파이프라인 구조

```
썸네일 이미지 (.jpg)
        │
        ▼
 [1단계] VLM (qwen2.5vl:7b)
        이미지를 보고 제목 텍스트 추출
        │
        ▼  (날 것의 OCR, 오탈자 포함 가능)
 [2단계] LLM (qwen2.5:7b)
        영상 제목/설명 맥락으로 오탈자 교정
        │
        ▼
 thumbnail_title 컬럼으로 저장
```

## 입출력

| 항목 | 경로 |
|---|---|
| 입력 CSV | `Data/SOCIETY/SOCIETY_new_category_v3.csv` |
| 출력 CSV | `Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv` |
| 체크포인트 | `Data/SOCIETY/ckpt_thumbnail_title.csv` |
| 기존 OCR 재활용 | `/home/urp_jwl2/urp_jungeun/data/SOCIETY_add_thumbnail_title.csv` (2,149건) |

## 사용법

```bash
# Ollama 서버 먼저 실행
ollama serve &

# 기본 실행 (qwen2.5vl:7b + 교정)
cd /home/urp_jwl2/26-1_URP
python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py"

# 고품질 모델 사용 (느리지만 정확)
python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py" --vlm-model qwen2.5vl:32b

# 교정 단계 생략 (빠른 실행)
python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py" --no-correct

# 체크포인트 저장 주기 변경 (기본 100건마다)
python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py" --batch-save 200
```

## 처리 대상

| 구분 | 건수 |
|---|---|
| 전체 썸네일 | 48,697개 |
| 기존 OCR 재활용 (urp_jungeun) | 2,149개 |
| 신규 추출 필요 | 46,548개 |

## 예상 소요 시간

| 모델 | 예상 시간 (RTX 4090 기준) |
|---|---|
| qwen2.5vl:7b (기본) | ~3~5시간 |
| qwen2.5vl:32b (고품질) | ~10~15시간 |
| qwen3-vl:32b (고품질) | ~10~15시간 |

## 재실행 (중단 후 이어서)

체크포인트가 자동 저장되므로 그냥 같은 명령을 다시 실행하면 됩니다.

```bash
python "3_썸네일 텍스트 추출 파이프라인/extract_thumbnail_title.py"
# → 체크포인트 로드 후 이어서 진행
```
