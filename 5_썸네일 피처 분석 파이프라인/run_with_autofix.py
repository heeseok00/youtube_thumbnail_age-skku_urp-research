"""
run_with_autofix.py
스크립트 실행 중 오류 발생 시 자동으로 패키지를 설치하고 재실행합니다.

사용법:
    python run_with_autofix.py 01_person_features.py
    python run_with_autofix.py 01_person_features.py --subcategory 정치/선거/시위
"""

import subprocess
import sys
import re
import argparse
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
MAX_RETRIES = 3

# 모듈명 → pip 패키지명 매핑
MODULE_TO_PACKAGE = {
    "ultralytics":    "ultralytics",
    "paddleocr":      "paddleocr",
    "paddle":         "paddlepaddle-gpu",
    "paddlepaddle":   "paddlepaddle-gpu",
    "mediapipe":      "mediapipe",
    "deepface":       "deepface",
    "tf_keras":       "tf-keras",
    "tensorflow":     "tensorflow",
    "cv2":            "opencv-python",
    "sklearn":        "scikit-learn",
    "scipy":          "scipy",
}

# 스크립트별 필수 패키지 (설치 필요 메시지 없이 미리 체크)
SCRIPT_REQUIREMENTS = {
    "01_person_features.py":    ["ultralytics"],
    "02_text_ratio.py":         ["paddlepaddle-gpu", "paddleocr"],
    "03_color_distribution.py": [],
    "05_gaze_direction.py":     ["mediapipe"],
    "06_facial_expression.py":  ["deepface", "tf-keras"],
}


def pip_install(package: str) -> bool:
    print(f"\n[자동 설치] pip install {package}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package, "-q"],
        capture_output=False
    )
    return result.returncode == 0


def extract_missing_module(stderr: str) -> str | None:
    """오류 메시지에서 누락된 모듈명 추출."""
    patterns = [
        r"No module named ['\"]([^'\"]+)['\"]",
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        r"\[설치 필요\] pip install ([\w\-]+)",
        r"ImportError: cannot import name .+ from ['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        m = re.search(pattern, stderr)
        if m:
            return m.group(1).split(".")[0]  # 서브모듈 제거 (e.g. paddle.fluid → paddle)
    return None


def run_script(script: str, extra_args: list) -> tuple[int, str]:
    """스크립트 실행 후 (returncode, stderr) 반환."""
    cmd = [sys.executable, str(PIPELINE_DIR / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"[실행] {script}  args={extra_args}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True,
                            stderr=subprocess.PIPE)
    return result.returncode, result.stderr or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="실행할 스크립트 파일명 (예: 01_person_features.py)")
    args, extra_args = parser.parse_known_args()

    script = args.script
    if not script.endswith(".py"):
        script += ".py"

    # 사전 필수 패키지 설치
    pre_reqs = SCRIPT_REQUIREMENTS.get(script, [])
    for pkg in pre_reqs:
        module = pkg.replace("-", "_").split("_gpu")[0]  # paddlepaddle-gpu → paddlepaddle
        try:
            __import__(module)
        except ImportError:
            pip_install(pkg)

    # 실행 + 자동 재시도
    for attempt in range(1, MAX_RETRIES + 1):
        returncode, stderr = run_script(script, extra_args)

        if returncode == 0:
            print(f"\n[완료] {script} 정상 종료")
            return

        print(f"\n[오류] 종료 코드: {returncode} (시도 {attempt}/{MAX_RETRIES})")

        missing = extract_missing_module(stderr)
        if missing:
            package = MODULE_TO_PACKAGE.get(missing, missing)
            print(f"[감지] 누락 모듈: {missing} → 패키지: {package}")
            success = pip_install(package)
            if success:
                print(f"[설치 완료] {package} → 재실행합니다...")
                continue
            else:
                print(f"[설치 실패] {package} 수동 설치 필요")
                break
        else:
            print("[알 수 없는 오류] 자동 수정 불가. 오류 내용:")
            print(stderr[-1000:])  # 마지막 1000자만 출력
            break

    print(f"\n[실패] {script} {MAX_RETRIES}회 시도 후 종료")
    sys.exit(1)


if __name__ == "__main__":
    main()
