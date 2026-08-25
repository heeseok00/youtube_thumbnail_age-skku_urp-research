# 7_HEALTH_다이어트_분석

HEALTH 카테고리 '다이어트' subcategory에 1~4단계 분석 파이프라인을 적용한다.

## 표본 (00_sample/)

- 원본: `Data/HEALTH/HEALTH_new_category_v3.csv` (60,506행, subcategory 분류 최종본)
- 다이어트 subcategory 11,113행 중, 영상별 **dominant 시청 연령대** 기준으로
  - 최댓값이 ~17 / 18~24 / 25~34 → `34-` 그룹 (풀 5,982개)
  - 최댓값이 65~ → `65+` 그룹 (풀 1,581개)
  - 35~64가 최댓값이면 제외, 연령 컬럼 결측 행 제외
- 각 그룹 **1,250개씩 무작위 추출** (`random_state=42` 고정, 총 2,500행)
- 별도 표본 CSV 없이 `00_sample/sampling.py`의 `load_diet_sample()`이 원본 v3에서
  매번 동일한 2,500행을 재현한다 (시드 고정 → 언제 돌려도 같은 표본)
  - 채널 편중 없음 (상위 5개 채널 점유율 34-: 1.8%, 65+: 3.6%)
  - 썸네일 파일 2,500/2,500 존재 확인됨
- **주의: 분석 기간 동안 원본 `HEALTH_new_category_v3.csv`를 수정하면 안 됨**
  (내용/행 순서가 바뀌면 같은 시드라도 다른 표본이 뽑힘)

각 단계 노트북에서 사용:

```python
import sys
sys.path.insert(0, "/home/urp_jwl/URP_backup/26-1_URP/7_HEALTH_다이어트_분석/00_sample")
from sampling import load_diet_sample
df = load_diet_sample()   # video_id, age_group, title, resolved_path(썸네일 절대경로) 등
```

## 단계별 폴더

| 폴더 | 내용 | 비고 |
|------|------|------|
| `01_gradcam/` | Grad-CAM (`gradcam_diet.ipynb`) | 실행 완료 (정확도 71.8%) |
| `02_image_caption/` | 이미지 캡션 분석 (LLaVA + XGBoost/SHAP) | 실행 완료 (정확도 62.2%, p=2.7e-08) |
| `03_title_analysis/` | 제목 분석 (피처 SHAP + RoBERTa/LIME) | 실행 완료 (XGB AUC 0.688, RoBERTa AUC 0.885) |
| `04_visual_features/` | 썸네일 시각 피처 분석 | 기존 `5_썸네일 피처 분석 파이프라인` 기반 튜닝 완료, 실행 대기 |

## 공통 규칙

1. 모든 단계의 입력은 `00_sample/sampling.py`의 `load_diet_sample()` 하나로 통일한다.
2. 각 단계 출력 CSV는 **`video_id` 컬럼을 유지**한다 (최종 병합 키).
3. 결과물은 각 단계 폴더의 `outputs/`에 저장한다.
4. 썸네일 경로는 반환 DataFrame의 `resolved_path` 컬럼(절대경로)을 그대로 사용한다.
