# 2.2 이미지 모델 test 예측 (전달용)

파일: `visual_test_predictions_videolevel.csv`

DINOv3 썸네일 분류기가 test 영상 1,884장에 대해 낸 65+ 확률입니다.

| 컬럼 | 의미 |
| --- | --- |
| video_id | 영상 ID |
| channel_id | 채널 ID |
| category | EDU / HEALTH / MEDITATION / SOCIETY (논문의 LIFESTYLE = MEDITATION) |
| y_true | 정답. 0 = ~34, 1 = 65+ |
| y_prob | 65+일 예측 확률 (0~1) |

## 분할

영상 단위 8:2, 연령 레이블 층화, random seed = 42.
같은 채널이 훈련과 검증에 함께 들어갑니다. 채널 단위로 다시 학습한 버전은 이 파일에 없습니다.

결합 실험에 쓰려면 텍스트 모델과 같은 채널 단위 split으로 재학습한 뒤 같은 형식의 csv를 다시 뽑아야 합니다.
