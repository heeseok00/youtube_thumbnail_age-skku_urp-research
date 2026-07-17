#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
LOG="$BASE/checkpoints/03b_full_run.log"

echo "=== 시작: $(date -Iseconds) ===" | tee "$LOG"
echo "대상: 시사/뉴스/사건 (체크포인트 있으면 이어서 실행)" | tee -a "$LOG"

"$PYTHON" "$BASE/03b_text_region_color.py" 2>&1 | tee -a "$LOG"

echo "=== merge: $(date -Iseconds) ===" | tee -a "$LOG"
"$PYTHON" "$BASE/merge_features.py" --quick 2>&1 | tee -a "$LOG"

echo "=== compare: $(date -Iseconds) ===" | tee -a "$LOG"
"$PYTHON" "$BASE/analysis/compare_age_groups.py" 2>&1 | tee -a "$LOG"

echo "=== 완료: $(date -Iseconds) ===" | tee -a "$LOG"
