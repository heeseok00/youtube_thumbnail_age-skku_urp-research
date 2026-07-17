"""
run_all.py
6개 피처 추출 파이프라인을 순차 실행합니다.

사용법:
    python run_all.py
    python run_all.py --steps 1 3 5       # 특정 스텝만 실행
    python run_all.py --subcategory 정치/선거/시위
    python run_all.py --input <csv경로> --output <결과csv경로>
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
DEFAULT_INPUT = str(PIPELINE_DIR.parent / "Data/SOCIETY/SOCIETY_new_category_v3_add_thumnail_title.csv")

STEPS = [
    (1, "01_person_features.py",    "인물 비중 + 인물 수 (통합)"),
    (2, "02_text_ratio.py",         "텍스트 비중"),
    (3, "03_color_distribution.py", "색상 분포 + 다양성"),
    (4, "03b_text_region_color.py", "텍스트/배경 ROI 색상"),
    (5, "04_gaze_direction.py",     "머리 방향 (head pose)"),
    (6, "05_facial_expression.py",  "인물 표정"),
]


def run_step(script: str, input_path: str, output_path: str, subcategory: str):
    cmd = [
        sys.executable,
        str(PIPELINE_DIR / script),
        "--input", input_path,
        "--output", output_path,
        "--subcategory", subcategory,
    ]
    print(f"\n{'='*60}")
    print(f"실행: {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[경고] {script} 실패 (종료 코드: {result.returncode})")
        return False
    return True


def main(args):
    selected_steps = set(args.steps) if args.steps else {s[0] for s in STEPS}
    current_input = args.input

    for step_num, script, label in STEPS:
        if step_num not in selected_steps:
            print(f"[건너뜀] Step {step_num}: {label}")
            continue

        # 중간 파일은 같은 경로에 덮어쓰거나 최종 output으로
        is_last = (step_num == max(selected_steps))
        output = args.output if (is_last and args.output) else current_input.replace(".csv", f"_feat{step_num}.csv")

        success = run_step(script, current_input, output, args.subcategory)
        if success:
            current_input = output  # 다음 스텝의 입력은 이전 스텝의 출력

    print(f"\n[완료] 최종 파일: {current_input}")
    print("분석을 실행하려면:")
    print(f"  python analysis/compare_age_groups.py --input {current_input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="썸네일 피처 전체 파이프라인 실행")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None, help="최종 결과 CSV 경로")
    parser.add_argument("--subcategory", default="시사/뉴스/사건")
    parser.add_argument("--steps", nargs="+", type=int, help="실행할 스텝 번호 (예: --steps 1 3 5)")
    main(parser.parse_args())
