# YouTube Thumbnail Age Analysis

> **성균관대학교 URP 2026 Spring** — 비전 임베딩을 활용하여 유튜브 썸네일의 연령대별 시각적 스타일 차이를 분석합니다.

## 프로젝트 개요

유튜브 영상의 썸네일 이미지만으로 타겟 연령대를 구분할 수 있을까요?  
사전학습된 비전 모델(DINOv2, SigLIP2)로 썸네일 임베딩을 추출하고, 해당 임베딩이 연령대를 얼마나 잘 분리하는지 정량적으로 평가합니다.

---

## 전체 파이프라인 흐름

```
00. 연령대 카테고리 선정
    ↓
01. 채널 목록 수집      vling.net 크롤링 → YT_channelsList_v1.csv
    ↓
02. 영상 데이터 수집     YouTube API → YT_dataset_v1.csv + thumbnails/
    ↓
03. 분석 파이프라인      임베딩 추출 → 평가 → 시각화
```

---

## 빠른 시작

```bash
# 1. 레포 클론
git clone https://github.com/heeseok00/youtube_thumbnail_age-skku_urp-research.git
cd youtube_thumbnail_age-skku_urp-research

# 2. 환경 복원
conda env create -f ytvenv.yaml
conda activate ytvenv

# 3. API 키 설정
echo "YOUTUBE_API_KEY=발급받은_키_입력" > .env

# 4. 노트북 순서대로 실행
jupyter notebook
```

> API 키 발급: [Google Cloud Console](https://console.cloud.google.com/) → API 및 서비스 → YouTube Data API v3 활성화

---

## 노트북 구성

| 번호 | 파일 | 설명 |
|---|---|---|
| 00 | `00.category_selection.ipynb` | 연령대 카테고리 기준 탐색 및 선정 |
| 01 | `01.channel_collection.ipynb` | vling.net에서 채널명·채널 ID 수집 → `YT_channelsList_v1.csv` |
| 02 | `02.data_collection.ipynb` | YouTube API로 영상 메타데이터·썸네일 수집 → `YT_dataset_v1.csv` |
| 03 | `03.thumbnail_age_pipeline.ipynb` | 임베딩 추출 / 모델 평가 / 시각화 |

---

## 데이터셋

### 레포에 포함된 파일

| 파일 | 설명 |
|---|---|
| `YT_channelsList_v1.csv` | 채널별 타겟 연령대 레이블 (수동 구성, 54개 채널) |
| `artifacts/` | 파이프라인 중간 결과물 (임베딩 메타데이터, 평가 점수, 시각화 이미지 등) |

`YT_channelsList_v1.csv` 구조:

```
channel_name, channel_id, target_age
Queen Solmee, UCahWSp7..., 18-24
...
```

연령대 6구간: `18-24` / `25-34` / `35-44` / `45-54` / `55-64` / `65+` (각 구간당 약 9개 채널)

### 직접 준비해야 하는 파일

| 파일 | 수집 방법 |
|---|---|
| `YT_dataset_v1.csv` | `02.data_collection.ipynb` 실행 |
| `thumbnails/` | `02.data_collection.ipynb` 실행 (자동 다운로드) |

> 채널 목록을 직접 구성하거나 확장하려면 `01.channel_collection.ipynb`을 먼저 실행하세요.

**연령대별 영상 수 (전체 270개 기준):**

| 18-24 | 25-34 | 35-44 | 45-54 | 55-64 | 65+ |
|---|---|---|---|---|---|
| 50 | 50 | 25 | 45 | 50 | 50 |

---

## 환경 설정

### 요구사항

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 Anaconda
- Python 3.14 (ytvenv.yaml에 명시)
- (선택) CUDA GPU — CPU로도 동작하나 임베딩 추출 속도 차이 있음

### conda 환경 복원

```bash
conda env create -f ytvenv.yaml
conda activate ytvenv
```

### API 키 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```
YOUTUBE_API_KEY=여기에_발급받은_키_입력
```

`.env`는 `.gitignore`에 포함되어 있어 레포에 올라가지 않습니다.

### 주요 패키지

| 패키지 | 용도 |
|---|---|
| `torch` + `transformers` | 비전 모델 추론 (DINOv2, SigLIP2) |
| `umap-learn` | 차원 축소 |
| `scikit-learn` | 평가 지표, t-SNE, 선형 탐침 |
| `playwright` | vling.net 채널 목록 크롤링 |
| `google-api-python-client` | YouTube Data API v3 |
| `python-dotenv` | `.env` API 키 로드 |
| `Pillow` + `numpy` | 이미지 로드 및 배열 연산 |
| `matplotlib` | 시각화 |
| `pandas` | 데이터 처리 |

---

## 분석 파이프라인 상세 (`03.thumbnail_age_pipeline.ipynb`)

```
Step 1    prepare    → Shorts 필터링, 연령대 레이블 병합
Step 1.5  download   → YouTube에서 썸네일 이미지 다운로드
Step 2    extract    → 비전 모델로 임베딩 추출
Step 3    evaluate   → 모델 간 클러스터링 성능 비교
Step 4    visualize  → UMAP / t-SNE 시각화 + 최근접 이웃 분석
```

### Step 1 — 데이터 준비

두 가지 기준으로 Shorts 영상을 필터링합니다:

| 기준 | 규칙 |
|---|---|
| 길이 | `duration ≤ 180초` |
| 키워드 | 제목 / 설명 / 태그에 `Shorts` 또는 `쇼츠` 포함 |

**결과:** 270개 전체 → 111개 Shorts 제거 → **159개** 분석 대상

### Step 2 — 임베딩 추출

HuggingFace 사전학습 모델로 L2 정규화된 이미지 임베딩을 추출합니다:

| Alias | 모델 | 임베딩 차원 |
|---|---|---|
| `dinov2-base` | `facebook/dinov2-base` | 768 |
| `siglip2-base` | `google/siglip2-base-patch16-224` | 768 |

> 모델은 최초 실행 시 HuggingFace Hub에서 자동 다운로드됩니다 (각 약 300~400MB).

### Step 3 — 모델 평가

UMAP으로 2차원 축소 후 네 가지 지표로 모델을 비교합니다:

| 지표 | 설명 |
|---|---|
| `silhouette_umap` | 클러스터 응집도·분리도 |
| `knn_purity_k10_umap` | k-NN 이웃 중 같은 연령대 비율 (k=10) |
| `trustworthiness_umap` | 원본 고차원 구조가 UMAP에 보존된 정도 |
| `linear_probe_macro_f1` | 선형 분류기 macro-F1 (3 seed 평균, 채널 그룹 분리) |

**평가 결과:**

| 모델 | Silhouette | kNN Purity k10 | Trustworthiness | Linear Probe F1 |
|---|---|---|---|---|
| **siglip2-base** ✓ | -0.014 | **0.522** | 0.858 | **0.375** |
| dinov2-base | -0.070 | 0.445 | **0.882** | 0.360 |

→ **SigLIP2-base** 선택

### Step 4 — 시각화

연령대별 색상으로 UMAP / t-SNE scatter plot을 생성하고, 최근접 이웃 분석표를 출력합니다.

`artifacts/visualizations/` 출력 파일:
- `umap_target_age.png`
- `tsne_target_age_perp{5,15,30}.png`
- `nearest_neighbor_examples.csv`

---

## 디렉터리 구조

```
.
├── 00.category_selection.ipynb    # 연령대 카테고리 선정
├── 01.channel_collection.ipynb    # vling.net 채널 목록 수집
├── 02.data_collection.ipynb       # YouTube API 영상 데이터 수집
├── 03.thumbnail_age_pipeline.ipynb # 임베딩 추출 / 평가 / 시각화
├── YT_channelsList_v1.csv         # 채널-연령대 레이블 (포함)
├── ytvenv.yaml                    # conda 환경 설정
├── .env                           # API 키 (gitignore, 직접 생성)
├── .gitignore
├── artifacts/
│   ├── dataset_with_flags.csv
│   ├── dataset_nonshort_final.csv
│   ├── embeddings/
│   │   ├── dinov2-base/           # metadata.csv, run_info.json
│   │   └── siglip2-base/          # metadata.csv, run_info.json
│   ├── evaluation/
│   │   ├── model_scores.csv
│   │   └── selected_model.json
│   └── visualizations/
│       ├── umap_target_age.png
│       └── nearest_neighbor_examples.csv
├── YT_dataset_v1.csv              # ← .gitignore (02번 노트북으로 수집)
└── thumbnails/                    # ← .gitignore (02번 노트북에서 자동 생성)
```

---

## 참고사항

- Shorts 필터링은 자동 판별만 적용된 상태입니다. `artifacts/manual_review_template.csv`를 작성하면 수동 검토 결과를 반영할 수 있습니다.
- 선형 탐침 평가에서 `GroupShuffleSplit`으로 같은 채널의 영상이 train/test에 동시에 포함되지 않도록 처리하여 데이터 누수를 방지합니다.
- SigLIP2는 이미지-텍스트 대조 학습 모델로, 구조상 텍스트 입력이 필요합니다. `"thumbnail"` 더미 텍스트를 자동 삽입하며, 실제 임베딩에는 `image_embeds`만 사용합니다.
