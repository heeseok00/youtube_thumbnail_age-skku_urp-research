#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/urp_jwl2/miniconda3/envs/urp/bin/python
BASE="/home/urp_jwl2/26-1_URP/5_썸네일 피처 분석 파이프라인"
LOG="$BASE/checkpoints/03b_full_run.log"

echo "=== 시작: $(date -Iseconds) ===" | tee "$LOG"
echo "대상: 시사/뉴스/사건 17,247건 (체크포인트 200건 제외, 약 17,047건 남음)" | tee -a "$LOG"
echo "예상 소요: 약 55~75분 (테스트 200건 기준 ~4.7 img/s)" | tee -a "$LOG"

"$PYTHON" "$BASE/03b_text_region_color.py" 2>&1 | tee -a "$LOG"

echo "=== merge: $(date -Iseconds) ===" | tee -a "$LOG"
"$PYTHON" "$BASE/merge_features.py" --quick 2>&1 | tee -a "$LOG"

echo "=== compare: $(date -Iseconds) ===" | tee -a "$LOG"
"$PYTHON" "$BASE/analysis/compare_age_groups.py" 2>&1 | tee -a "$LOG"

echo "=== 완료: $(date -Iseconds) ===" | tee -a "$LOG"
