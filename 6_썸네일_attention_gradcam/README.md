# 썸네일 Attention / GradCAM 분석 (시사·뉴스·사건)

Society 카테고리 중 **시사/뉴스/사건** 서브카테고리만 대상으로 DINOv2 attention map·ROI deletion/insertion 분석.

## 파일

| 파일 | 설명 |
|------|------|
| `attention_gradcam_news.ipynb` | 메인 노트북 (Jupyter에서 Run All) |
| `build_news_merged_csv.py` | merged CSV만 미리 생성 |
| `data/SOCIETY_news_with_dinov2.csv` | §2 실행 시 자동 생성 |
| `outputs/` | top50 CSV, 히트맵, ROI 캐시 |

## 사전 요구

- GPU + CUDA (`device=cuda`)
- **권장 conda 환경:** `ytvenv`

```bash
conda activate ytvenv
cd /path/to/26-1_URP/6_썸네일_attention_gradcam
```

### ytvenv 주의 (transformers ↔ PyTorch)

`ytvenv`에 **transformers 5.5** + **torch 2.2** 조합이면 DINOv2 로드가 실패할 수 있습니다.  
노트북 실행 전 한 번만 확인:

```bash
conda activate ytvenv
python -c "from transformers.utils import is_torch_available; print('torch OK:', is_torch_available())"
```

`False`이면:

```bash
pip install 'transformers>=4.36,<4.46'
```

(현재 서버에서는 `transformers 4.45.2` + `torch 2.2.2`로 DINOv2 로드 확인됨)

Jupyter 커널도 **Python 3.10 (ytvenv)** 로 선택해야 합니다.

## 실행 순서

```bash
conda activate ytvenv
cd /path/to/26-1_URP/6_썸네일_attention_gradcam

# (선택) merged CSV만 먼저 만들기
python build_news_merged_csv.py

# Jupyter — 커널: ytvenv
jupyter notebook attention_gradcam_news.ipynb
# → Run All
```

## 설정 (노트북 §1)

경로는 절대경로가 아니라 **상대경로**로 잡습니다 (`26-1_URP`와 형제인 `urp_bin/`).

- **입력:** `26-1_URP/Data/SOCIETY/SOCIETY_new_category_v3.csv`
- **DINOv2 피처:** `urp_bin/SOCIETY 파일들 - 썸네일 사진 분석/SOCIETY_final_kr_clean_with_dinov2_768_fixed.csv`
- **썸네일:** `26-1_URP/Data/SOCIETY/thumbnails/`
- **서브카테고리:** `시사/뉴스/사건`

## 원본 코드

프로토타입: `urp_bin/SOCIETY 파일들 - 썸네일 사진 분석/attention_gradcam_style_38_606_108.ipynb`

## 주의

- `outputs/`를 비우거나 새로 Run All 하면 top50 CSV가 **시사/뉴스/사건** 기준으로 다시 생성됩니다.
- 예전 `urp_bin/attn_gradcam_outputs/` 결과는 Society **전체** 기준이므로 혼동하지 마세요.
