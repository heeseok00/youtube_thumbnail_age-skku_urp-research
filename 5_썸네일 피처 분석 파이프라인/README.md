# 5_썸네일 피처 분석 파이프라인

SOCIETY '시사/뉴스/사건' subcategory 썸네일에서 6개 피처를 추출하고,
~34세 이하 vs 65세 이상 시청자 집단 간 비교 분석합니다.

## 구조

```
5_썸네일 피처 분석 파이프라인/
├── utils.py                    공통 유틸 (경로 변환, 체크포인트, CSV 로딩)
├── 01_person_features.py       인물 비중 + 인물 수 (YOLOv8-seg, 통합)
├── 02_text_ratio.py            텍스트 비중 (EasyOCR)
├── 03_color_distribution.py    색상 분포 (HSV, PIL)
├── 03b_text_region_color.py    텍스트/배경 ROI 색상 (EasyOCR)
├── 04_gaze_direction.py        머리 방향 / head pose (MediaPipe)
├── 05_facial_expression.py     인물 표정 (FER)
├── run_all.py                  전체 파이프라인 순차 실행
├── merge_features.py           체크포인트 → merged CSV 병합
├── checkpoints/                중간 저장 (자동 생성)
└── analysis/
    ├── compare_age_groups.py   통계 분석 + 시각화
    ├── multivariate_analysis.py 상관관계 + RF 중요도
    ├── visualize_per_feature.py 피처별 개별 시각화
    └── outputs/                결과 이미지/CSV (자동 생성)
```

## 의존성 설치

```bash
# 필수
pip install ultralytics          # 01 (YOLOv8)
pip install easyocr              # 02, 03b
pip install mediapipe            # 04
pip install fer opencv-python-headless  # 05

# 이미 설치됨
# torch, numpy, pandas, pillow, scipy, matplotlib, seaborn, scikit-learn
```

## 실행 방법

### 전체 파이프라인 한 번에
```bash
cd /path/to/26-1_URP
python "5_썸네일 피처 분석 파이프라인/run_all.py"
```

### 특정 스텝만
```bash
python "5_썸네일 피처 분석 파이프라인/run_all.py" --steps 1 3  # 인물 + 색상만
```

### 개별 스크립트 실행
```bash
python "5_썸네일 피처 분석 파이프라인/01_person_features.py"
python "5_썸네일 피처 분석 파이프라인/03_color_distribution.py"  # 추가 설치 불필요
```

### 분석 실행 (피처 추출 후)
```bash
python "5_썸네일 피처 분석 파이프라인/analysis/compare_age_groups.py" --input <피처CSV경로>
```

## 피처별 출력 컬럼

| 스크립트 | 출력 컬럼 | 타입 |
|---|---|---|
| 01 | `person_ratio`, `person_count`, `person_count_cat` | float / int / str |
| 02 | `text_ratio` | float (0~1) |
| 03 | `color_hue_mean`, `color_saturation`, `color_brightness`, `color_warm_ratio`, `color_entropy`, `color_hue_std` | float |
| 03b | `text_color_*`, `bg_color_entropy`, `text_bg_*_diff` | float |
| 04 | `head_pose`, `head_pose_face_count` | str / int |
| 05 | `expression_dominant`, `expression_happy`, `expression_angry`, `expression_neutral`, `expression_sad` | str / float |

## 참고

- 모든 스크립트는 체크포인트 자동 감지 — 중단 후 재실행 시 이어서 진행
- 다른 subcategory 적용: `--subcategory 정치/선거/시위`
- 다른 CSV 적용: `--input <경로>`
- 경로: `utils.py`의 `BASE_DIR`은 스크립트 위치 기준 상대경로라 계정에 무관하게 동작
