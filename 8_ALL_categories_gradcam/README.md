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
```

## 산출물 (`outputs/`)
- `all_sample.csv`, `train_split.csv`, `test_split.csv`
- `dinov3_classifier.pt`, `metrics.json`
- `gradcam_correct_{34,65}_top50.png`, `del_ins_*.png`, `roi_analysis_*.png` 등
