# YouTube Thumbnail Age Analysis

> **성균관대학교 URP 2026 Spring** — 비전 임베딩을 활용하여 유튜브 썸네일의 연령대별 시각적 스타일 차이를 분석합니다.

## 프로젝트 개요

유튜브 영상의 썸네일 이미지만으로 타겟 연령대를 구분할 수 있을까요?  
사전학습된 비전 모델(DINOv2, SigLIP2)로 썸네일 임베딩을 추출하고, 해당 임베딩이 연령대를 얼마나 잘 분리하는지 정량적으로 평가합니다.

## 데이터셋

| 파일 | 설명 |
|---|---|
| `YT_dataset_v1.csv` | 영상별 메타데이터 + 썸네일 경로 (270개 영상, 54개 채널) |
| `YT_channelsList_v1.csv` | 채널별 타겟 연령대 레이블 (`18-24` ~ `65+`) |

**연령대별 영상 수 (전체):**

| 18-24 | 25-34 | 35-44 | 45-54 | 55-64 | 65+ |
|---|---|---|---|---|---|
| 50 | 50 | 25 | 45 | 50 | 50 |

## 파이프라인

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

`is_short_auto` → `is_short_manual` → `is_short_final` 3단계 플래그 구조로, 수동 검토를 통한 정제도 지원합니다.

**결과:** 270개 전체 → 111개 Shorts 제거 → **159개** 분석 대상

### Step 1.5 — 썸네일 다운로드

CSV의 `thumbnail_url`에서 이미지를 받아 `thumbnail_path` 위치에 저장합니다.  
이미 다운로드된 파일은 건너뛰므로 중단 후 재실행해도 안전합니다.

### Step 2 — 임베딩 추출

HuggingFace 사전학습 모델로 L2 정규화된 이미지 임베딩을 추출합니다:

| Alias | 모델 | 임베딩 차원 |
|---|---|---|
| `dinov2-base` | `facebook/dinov2-base` | 768 |
| `siglip2-base` | `google/siglip2-base-patch16-224` | 768 |

모델별 출력: `embeddings.npy` (N × 768), `metadata.csv`, `run_info.json`

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

## 디렉터리 구조

```
.
├── 채널_시각화.ipynb              # 메인 파이프라인 노트북
├── ytvenv.yaml                    # conda 환경 설정
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
└── (데이터 파일은 .gitignore로 제외)
```

> `embeddings.npy`, `thumbnails/`, 원본 CSV 데이터셋은 레포에 포함되지 않습니다.

## 환경 설정

**1. conda 환경 복원**

```bash
conda env create -f ytvenv.yaml
conda activate ytvenv
```

**2. 프로젝트 루트에 데이터 파일 배치**

```
YT_dataset_v1.csv
YT_channelsList_v1.csv
```

**3. 노트북 셀 순서대로 실행**

```
설정 셀 → Step 1 → Step 1.5 → Step 2 → Step 3 → Step 4
```

## 주요 의존성

| 패키지 | 용도 |
|---|---|
| `torch` + `transformers` | 비전 모델 추론 |
| `umap-learn` | 차원 축소 |
| `scikit-learn` | 평가 지표, t-SNE, 선형 탐침 |
| `Pillow` + `numpy` | 이미지 로드 및 배열 연산 |
| `matplotlib` | 시각화 |
| `pandas` | 결과 확인 |

## 참고사항

- Shorts 필터링은 자동 판별만 적용된 상태입니다 (`manual_labels = 0`). `artifacts/manual_review_template.csv`를 작성하면 수동 검토 결과를 반영할 수 있습니다.
- 선형 탐침 평가에서 `GroupShuffleSplit`으로 같은 채널의 영상이 train/test에 동시에 포함되지 않도록 처리하여 데이터 누수를 방지합니다.
- SigLIP2는 이미지-텍스트 대조 학습 모델로, 구조상 텍스트 입력이 필요합니다. `"thumbnail"` 더미 텍스트를 자동 삽입하며, 실제 임베딩에는 `image_embeds`만 사용합니다.
