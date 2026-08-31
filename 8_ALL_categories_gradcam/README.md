# 8_ALL_categories_gradcam (DINOv2)

4카테고리(SOCIETY / EDU / HEALTH다이어트 / MEDITATION) 통합 표본으로
**한 모델**을 학습하고 Grad-CAM + ROI(del/ins)까지 실행합니다.

## 백본
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (현재 사용, HF gated)

## 실행
```bash
cd /home/urp_jwl/URP_backup/26-1_URP/8_ALL_categories_gradcam
conda run -n urp_yena python prepare_data.py
conda run -n urp_yena python run_gradcam_pipeline.py --stage all
# top-50 히트맵을 text/person/background 에너지 비율로만 다시 계산
conda run -n urp_yena python run_gradcam_pipeline.py --stage camshare
# 논문 그림 (본문 Figure 1 + 부록 Figure A2/A3)
conda run -n urp_yena python make_paper_figures.py
```

## 산출물 (`outputs/`)
- `all_sample.csv`, `train_split.csv`, `test_split.csv`
- `dinov3_classifier.pt`, `metrics.json`
- `gradcam_correct_{34,65}_top50.png`, `del_ins_*.png`, `roi_analysis_*.png` 등
- `gradcam_fig1_{34,65}.png`, `samples_fig1_{34,65}.csv` — 논문 본문 그림 1 (카테고리당 1장)
- `gradcam_appendix_{34,65}_12.png`, `samples_appendix_{34,65}_12.csv` — 논문 부록 그림 A2/A3 (카테고리당 3장)
- `cam_roi_share_per_image.csv`, `cam_roi_share_summary.csv`, `cam_roi_share_bars.png`
  — 예측 클래스 Grad-CAM이 text / person / background에 모인 비율
  (`share` = 히트맵 에너지, `hot_share` = CAM≥0.5 픽셀, `concentration` = share/area)
