"""
run_all.py
4단계 시각 피처 파이프라인을 순차 실행 (추출 → 병합 → 분석).

사용법:
    python run_all.py                 # 전체 실행
    python run_all.py --steps 1 3     # 특정 추출 스텝만
    python run_all.py --skip-analysis # 추출·병합만
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

EXTRACT_STEPS = [
    (1, "01_person_features.py",    "인물 비중 + 인물 수 (YOLOv8-seg)"),
    (2, "02_text_ratio.py",         "텍스트 비중 (EasyOCR)"),
    (3, "03_color_distribution.py", "색상 분포 (CPU)"),
    (4, "03b_text_region_color.py", "텍스트/배경 ROI 색상 (EasyOCR)"),
    (5, "04_gaze_direction.py",     "머리 방향 (MediaPipe)"),
    (6, "05_facial_expression.py",  "인물 표정 (FER)"),
]

POST_STEPS = [
    ("06_merge_features.py",      "체크포인트 병합"),
    ("07_compare_age_groups.py",  "연령대 비교 분석"),
    ("08_multivariate_analysis.py", "상관관계 + RF 중요도"),
]


def run_script(script: str, extra_args=None) -> bool:
    cmd = [sys.executable, str(PIPELINE_DIR / script)] + (extra_args or [])
    print(f"\n{'='*60}\n실행: {script}\n{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[경고] {script} 실패 (종료 코드: {result.returncode})")
        return False
    return True


def main(args):
    selected = set(args.steps) if args.steps else {s[0] for s in EXTRACT_STEPS}

    for step_num, script, label in EXTRACT_STEPS:
        if step_num not in selected:
            print(f"[건너뜀] Step {step_num}: {label}")
            continue
        extra = ["--test", str(args.test)] if args.test > 0 else None
        run_script(script, extra)

    if not args.skip_analysis:
        for script, label in POST_STEPS:
            run_script(script)

    print("\n[완료] 결과: outputs/diet_visual_features.csv, outputs/analysis/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다이어트 시각 피처 전체 파이프라인")
    parser.add_argument("--steps", nargs="+", type=int, help="실행할 추출 스텝 번호 (예: --steps 1 3)")
    parser.add_argument("--test", type=int, default=0, help="테스트: 스텝당 N개만 처리")
    parser.add_argument("--skip-analysis", action="store_true", help="병합·분석 건너뜀")
    main(parser.parse_args())
