# YouTube Thumbnail — 카테고리·채널 임베딩 분석

> **성균관대학교 URP 2026 Spring**  
> 썸네일 이미지를 **DINOv2-base**, **SigLIP2-base**로 임베딩하고, **채널 단위**로 묶은 뒤 **UMAP / t-SNE** 및 **K-means**로 구조를 탐색합니다.

---

## 프로젝트 개요

- **입력:** 카테고리별 영상·썸네일 메타데이터 CSV (`YT_dataset_{카테고리}.csv`) 및 로컬 썸네일 경로  
- **임베딩:** 영상(썸네일) 단위 768차원 벡터 → 동일 `channel_id`끼리 **평균** 후 **L2 정규화** → 채널당 1벡터  
- **시각화:** 채널 벡터에 대한 UMAP·t-SNE(03), K-means 라벨을 색으로 입힌 UMAP(04)  
- **카테고리(5개):** `health`, `food`, `VLOG`, `society`, `education`

---

## 전체 파이프라인

```
00. 카테고리·기준 탐색          → 00.category_selection.ipynb
01. 채널 후보 수집              → 01.channel_collection.ipynb
02. YouTube API 데이터·썸네일   → 02.data_collection.py (+ .env API 키)
03. 임베딩·채널 평균·2D 시각화  → 03.thumbnail_age_pipeline.ipynb
04. 고차원 K-means·UMAP(색)     → 04_channel_clustering.ipynb
```

배치로 5개 카테고리 수집을 연속 실행하려면 [`yt_all_collect.bat`](yt_all_collect.bat)을 사용합니다 (내부에서 `02.data_collection.py` 호출).

---

## 빠른 시작

```bash
git clone https://github.com/heeseok00/youtube_thumbnail_age-skku_urp-research.git
cd youtube_thumbnail_age-skku_urp-research

conda env create -f ytvenv.yaml
conda activate ytvenv
```

프로젝트 루트에 `.env`를 두고 YouTube API 키를 넣습니다 (`.gitignore` 대상).

```env
YOUTUBE_API_KEY=발급받은_키
```

- **GPU:** PyTorch CUDA 빌드 사용 시 임베딩(03 Step 2)이 빠릅니다. `03` 설정 셀에서 `DEVICE = "cuda"` 등으로 지정.  
- **Hugging Face:** 모델 다운로드 시 rate limit 완화를 위해 `hf auth login` 또는 `HF_TOKEN` 권장.  
- **Windows:** OpenMP 충돌 완화를 위해 노트북에 `KMP_DUPLICATE_LIB_OK` 설정이 들어가 있습니다. K-means 경고 완화는 `04` 설정 셀의 `OMP_NUM_THREADS`를 참고하세요.

---

## 주요 파일

| 구분 | 파일 | 설명 |
|------|------|------|
| 노트북 | `00.category_selection.ipynb` | 카테고리·선정 메모 |
| 노트북 | `01.channel_collection.ipynb` | 채널 목록 수집 |
| 스크립트 | `02.data_collection.py` | YouTube Data API v3로 채널별 영상 메타·썸네일 다운로드 |
| 배치 | `yt_all_collect.bat` | 카테고리별 `02` 실행 예시 |
| 노트북 | `03.thumbnail_age_pipeline.ipynb` | 준비 → 임베딩 → 채널 평균 → UMAP/t-SNE |
| 노트북 | `04_channel_clustering.ipynb` | 채널 벡터 K-means(실루엣 k 탐색) + UMAP 색상 플롯 |
| 환경 | `ytvenv.yaml` | conda 환경 고정 (PyTorch cu128 등은 주석·공식 안내 참고) |

### 데이터 CSV (예시 이름)

| 파일 | 설명 |
|------|------|
| `YT_ChannelData_{카테고리}_clean.csv` | 수집에 쓰는 채널 목록(정리본) |
| `YT_dataset_{카테고리}.csv` | 영상·썸네일 경로 등 메인 데이터셋 |
| `artifacts/{카테고리}/dataset_ready.csv` | 03 Step 1에서 썸네일 파일이 실제 존재하는 행만 필터 |

썸네일 이미지 폴더(`thumbnails_*` 등)는 용량이 커서 보통 Git에 포함하지 않습니다 (`.gitignore` 참고).

---

## `03.thumbnail_age_pipeline.ipynb` 단계

| 단계 | 내용 | 산출(요약) |
|------|------|------------|
| Step 1 | `prepare` | `artifacts/{cat}/dataset_ready.csv` |
| Step 2 | `extract` | `embeddings/{dinov2-base\|siglip2-base}/embeddings.npy`, `metadata.csv` |
| Step 3 | `aggregate` | `channel_embeddings.npy`, `channel_metadata.csv` (채널 평균 + L2 정규화) |
| Step 4 | `visualize` | `visualizations/.../umap_channels.png`, `tsne_channels_perp*.png` |

- 임베딩은 **고차원에서** 채널 평균을 낸 뒤 정규화합니다.  
- SigLIP2는 이미지 전용 경로(`get_image_features` 등)로 처리합니다.

---

## `04_channel_clustering.ipynb`

- 입력: `channel_embeddings.npy` + `channel_metadata.csv` (행 순서 일치)  
- `K_MIN`~`K_MAX` 범위에서 **실루엣(euclidean)** 이 가장 큰 **k**를 선택해 K-means 라벨 저장  
- 산출(모델 폴더별): `kmeans_labels_k{k}.csv`, `kmeans_sweep.json`, `kmeans_summary.json`  
- 선택: `visualizations/{모델}/umap_kmeans_k{k}.png` (같은 채널 벡터로 UMAP 후 라벨 색상)

---

## `artifacts/` 디렉터리 구조 (요약)

```
artifacts/{카테고리}/
├── dataset_ready.csv
├── embeddings/{dinov2-base|siglip2-base}/
│   ├── embeddings.npy              # Git 제외 (.gitignore)
│   ├── metadata.csv
│   ├── channel_embeddings.npy      # Git 제외
│   ├── channel_metadata.csv
│   ├── kmeans_labels_k*.csv        # 04 실행 시
│   ├── kmeans_sweep.json
│   └── kmeans_summary.json
└── visualizations/{dinov2-base|siglip2-base}/
    ├── umap_channels.png
    ├── tsne_channels_perp*.png
    └── umap_kmeans_k*.png          # 04 실행 시
```

대용량 `.npy`는 `.gitignore`로 제외하고, CSV·JSON·PNG 등만 저장소에 올리는 구성을 권장합니다.

---

## `02.data_collection.py` 실행 예

```bash
conda activate ytvenv
python 02.data_collection.py --channels-csv YT_ChannelData_Health_clean.csv --output-csv YT_dataset_health.csv --video-count 10
```

옵션은 `python 02.data_collection.py --help`로 확인합니다.

---

## 참고 링크

- [YouTube Data API v3](https://developers.google.com/youtube/v3)  
- [PyTorch 설치 (CUDA)](https://pytorch.org/get-started/locally/)  
- [Hugging Face Hub — 로그인 / 토큰](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)

---

## 라이선스·저장소

원격 저장소: `https://github.com/heeseok00/youtube_thumbnail_age-skku_urp-research`  

연구/URP 보고용 산출물과 경로는 팀 정책에 맞게 조정하세요.
