# 04_visual_features — 시각 피처 분석 (4단계)

다이어트 샘플 2,500개(34- 1,250 / 65+ 1,250) 썸네일에서 시각 피처 6종을 추출하고
연령대 간 비교 분석합니다.
(원본: `5_썸네일 피처 분석 파이프라인` — 데이터 로딩을 `00_sample/sampling.py` 공유 모듈로 교체)

## 구조

```
04_visual_features/
├── utils.py                    샘플 로딩 + 체크포인트 유틸
├── 01_person_features.py       인물 비중 + 인물 수 (YOLOv8s-seg, GPU)
├── 02_text_ratio.py            텍스트 비중 (EasyOCR, GPU)
├── 03_color_distribution.py    색상 분포 (PIL/numpy, CPU)
├── 03b_text_region_color.py    텍스트/배경 ROI 색상 (EasyOCR, GPU)
├── 04_gaze_direction.py        머리 방향 head pose (MediaPipe, CPU)
├── 05_facial_expression.py     인물 표정 (FER, CPU/GPU)
├── 06_merge_features.py        체크포인트 → 최종 피처 CSV 병합
├── 07_compare_age_groups.py    Mann-Whitney/Chi-square 비교 + 시각화
├── 08_multivariate_analysis.py Spearman 상관 + RF 피처 중요도
├── run_all.py                  전체 순차 실행
├── checkpoints/                중간 저장 (자동 생성, git 미추적)
└── outputs/                    최종 CSV·그림 (자동 생성, git 미추적)
```

## 실행

```bash
conda activate urp_yena
cd "26-1_URP/7_HEALTH_다이어트_분석/04_visual_features"

# 스모크 테스트 (스텝당 20개)
python run_all.py --test 20 --skip-analysis

# 전체 실행 (추출 → 병합 → 분석)
python run_all.py
```

- 모든 스크립트는 체크포인트를 자동 감지하므로 중단 후 재실행하면 이어서 진행됩니다.
- `05_facial_expression.py`는 `04_gaze_direction.py`의 체크포인트가 있으면
  얼굴 없는 썸네일을 자동 스킵하므로 04 → 05 순서를 권장합니다.
- GPU 사용 스텝(01, 02, 03b)은 1~3단계 GPU 작업과 겹치지 않게 실행하세요.

## 출력

| 파일 | 내용 |
|---|---|
| `outputs/diet_visual_features.csv` | 샘플 + 전체 피처 (약 2,422행) |
| `outputs/analysis/01_continuous_features.png/.csv` | 연속형 피처 violin + Mann-Whitney |
| `outputs/analysis/02_categorical_features.png/.csv` | 범주형 피처 bar + Chi-square |
| `outputs/analysis/03_text_region_color.png/.csv` | 텍스트 ROI 색상 비교 |
| `outputs/analysis/corr_matrix.png` | Spearman 상관 히트맵 |
| `outputs/analysis/rf_feature_importance.png/.csv` | RF 피처 중요도 (5-fold CV) |
