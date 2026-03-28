# YouTube Thumbnail Age Analysis

> **SKKU URP 2026 Spring** — Analyzing visual style differences in YouTube thumbnails across target audience age groups using vision embeddings.

## Overview

Can we identify the target age group of a YouTube video from its thumbnail image alone?  
This project builds a pipeline that extracts visual embeddings from thumbnail images using pretrained vision models (DINOv2, SigLIP2) and evaluates how well those embeddings separate audience age groups in embedding space.

## Dataset

| File | Description |
|---|---|
| `YT_dataset_v1.csv` | Per-video metadata + thumbnail local paths (270 videos, 54 channels) |
| `YT_channelsList_v1.csv` | Per-channel target age label (`18-24` ~ `65+`) |

**Age group distribution (all videos):**

| 18-24 | 25-34 | 35-44 | 45-54 | 55-64 | 65+ |
|---|---|---|---|---|---|
| 50 | 50 | 25 | 45 | 50 | 50 |

## Pipeline

```
Step 1   prepare     → Shorts filtering, label merge
Step 1.5 download    → Fetch thumbnail images from YouTube
Step 2   extract     → Vision model embedding extraction
Step 3   evaluate    → Clustering quality metrics across models
Step 4   visualize   → UMAP / t-SNE plots + nearest-neighbor inspection
```

### Step 1 — Prepare

Filters YouTube Shorts from the dataset using two criteria:

| Criterion | Rule |
|---|---|
| Duration | `duration ≤ 180s` |
| Keyword | title / description / tags contain `Shorts` or `쇼츠` |

Outputs a three-stage flag: `is_short_auto` → `is_short_manual` → `is_short_final`

**Results:** 270 total → 111 Shorts filtered → **159 videos** for analysis

### Step 1.5 — Download Thumbnails

Downloads thumbnail images from `thumbnail_url` column into `thumbnail_path` locations.  
Already-downloaded files are skipped, so re-running is safe.

### Step 2 — Extract Embeddings

Extracts L2-normalized image embeddings using HuggingFace models:

| Alias | Model | Embedding Dim |
|---|---|---|
| `dinov2-base` | `facebook/dinov2-base` | 768 |
| `siglip2-base` | `google/siglip2-base-patch16-224` | 768 |

Outputs per model: `embeddings.npy` (N × 768), `metadata.csv`, `run_info.json`

### Step 3 — Evaluate

Compares models on four metrics computed in UMAP-projected 2D space:

| Metric | Description |
|---|---|
| `silhouette_umap` | Cluster cohesion and separation |
| `knn_purity_k10_umap` | Fraction of same-age neighbors in k-NN (k=10) |
| `trustworthiness_umap` | How well UMAP preserves original structure |
| `linear_probe_macro_f1` | Macro-F1 of linear classifier (3-seed avg, channel-group split) |

**Results:**

| Model | Silhouette | kNN Purity k10 | Trustworthiness | Linear Probe F1 |
|---|---|---|---|---|
| **siglip2-base** ✓ | -0.014 | **0.522** | 0.858 | **0.375** |
| dinov2-base | -0.070 | 0.445 | **0.882** | 0.360 |

→ **SigLIP2-base** selected as the best model.

### Step 4 — Visualize

Generates UMAP and t-SNE scatter plots colored by age group, and a nearest-neighbor inspection table.

Output files in `artifacts/visualizations/`:
- `umap_target_age.png`
- `tsne_target_age_perp{5,15,30}.png`
- `nearest_neighbor_examples.csv`

## Repository Structure

```
.
├── 채널_시각화.ipynb          # Main pipeline notebook
├── ytvenv.yaml                # Conda environment spec
├── .gitignore
├── artifacts/
│   ├── dataset_with_flags.csv
│   ├── dataset_nonshort_final.csv
│   ├── embeddings/
│   │   ├── dinov2-base/       # metadata.csv, run_info.json
│   │   └── siglip2-base/      # metadata.csv, run_info.json
│   ├── evaluation/
│   │   ├── model_scores.csv
│   │   └── selected_model.json
│   └── visualizations/
│       ├── umap_target_age.png
│       └── nearest_neighbor_examples.csv
└── (data files excluded — see .gitignore)
```

> `embeddings.npy`, `thumbnails/`, and raw CSV datasets are excluded from the repository.

## Setup

**1. Restore the conda environment**

```bash
conda env create -f ytvenv.yaml
conda activate ytvenv
```

**2. Place data files in the project root**

```
YT_dataset_v1.csv
YT_channelsList_v1.csv
```

**3. Run the notebook cells in order**

```
설정 셀 → Step 1 → Step 1.5 → Step 2 → Step 3 → Step 4
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` + `transformers` | Vision model inference |
| `umap-learn` | Dimensionality reduction |
| `scikit-learn` | Evaluation metrics, t-SNE, linear probe |
| `Pillow` + `numpy` | Image loading and array ops |
| `matplotlib` | Visualization |
| `pandas` | Result inspection |

## Notes

- Shorts filtering uses auto-detection only (`manual_labels = 0`). Manual review via `artifacts/manual_review_template.csv` is supported.
- Channel-group train/test split (`GroupShuffleSplit`) is used in the linear probe to prevent data leakage across videos from the same channel.
- SigLIP requires dummy text input (`"thumbnail"`) alongside images due to its dual-encoder architecture; only `image_embeds` are used as the final embedding.
