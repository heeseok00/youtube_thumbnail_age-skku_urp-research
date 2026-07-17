"""
merge_features.py
완료된 피처 체크포인트들을 merged CSV에 반영합니다.
feat05(expression) 완료 대기 후 자동 실행됩니다.
"""

import time
import pandas as pd
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / "Data/SOCIETY"
PREFIX  = "SOCIETY_new_category_v3_add_thumnail_title"
MERGED  = DATA / f"{PREFIX}_features_merged.csv"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

EXPR_CKPT = CKPT_DIR / "06_facial_expression.csv"
TEXT_CKPT = CKPT_DIR / "02_text_ratio.csv"
TEXT_COLOR_CKPT = CKPT_DIR / "03b_text_region_color.csv"
TEXT_COLOR_COLS = [
    "text_color_entropy", "text_color_saturation", "text_color_brightness",
    "text_color_hue_std", "bg_color_entropy", "text_bg_entropy_diff",
    "text_bg_saturation_diff",
]


EXPR_TOTAL = 17_247  # 시사/뉴스/사건 전체 영상 수

def wait_for_expression(check_interval=60):
    """expression 체크포인트 행 수가 전체 대상(17,247)의 99% 이상 될 때까지 대기.
    얼굴 없는 영상은 expression_dominant가 NaN이므로 non-null 비율 대신 행 수로 판단."""
    print("[대기] feat05(표정) 완료 대기 중...")
    while True:
        if EXPR_CKPT.exists():
            df = pd.read_csv(EXPR_CKPT)
            total_rows = len(df)
            done = df["expression_dominant"].notna().sum()
            print(f"  표정 진행: {total_rows:,} / {EXPR_TOTAL:,} 행 처리 (표정 추출: {done:,}건)")
            if total_rows >= EXPR_TOTAL * 0.99:  # 전체 행 수 기준 99% 이상
                print("[완료] 표정 추출 완료 확인")
                return df
        time.sleep(check_interval)


def merge_expression(merged_df, expr_df):
    expr_cols = ["video_id", "expression_dominant", "expression_happy",
                 "expression_angry", "expression_neutral", "expression_sad"]
    for col in expr_cols[1:]:
        merged_df = merged_df.drop(columns=[col], errors="ignore")
    merged_df = merged_df.merge(
        expr_df[expr_cols].dropna(subset=["expression_dominant"]),
        on="video_id", how="left"
    )
    return merged_df


def merge_text_region_color(merged_df):
    if not TEXT_COLOR_CKPT.exists():
        print("[스킵] text_region_color 체크포인트 없음")
        return merged_df
    color_df = pd.read_csv(TEXT_COLOR_CKPT)
    done = color_df["text_color_entropy"].notna().sum()
    print(f"[text_region_color] 텍스트 ROI 완료: {done:,} / {len(color_df):,}")
    if len(color_df) < 10:
        print("[스킵] text_region_color 데이터 부족")
        return merged_df
    for col in TEXT_COLOR_COLS:
        merged_df = merged_df.drop(columns=[col], errors="ignore")
    merged_df = merged_df.merge(color_df[["video_id"] + TEXT_COLOR_COLS], on="video_id", how="left")
    return merged_df


def merge_text_ratio(merged_df):
    if not TEXT_CKPT.exists():
        print("[스킵] text_ratio 체크포인트 없음")
        return merged_df
    text_df = pd.read_csv(TEXT_CKPT)
    done = text_df["text_ratio"].notna().sum()
    print(f"[text_ratio] 현재 완료: {done:,} / 17,247 ({done/17247*100:.1f}%)")
    if done < 100:
        print("[스킵] text_ratio 데이터 부족 (100건 미만)")
        return merged_df
    merged_df = merged_df.drop(columns=["text_ratio"], errors="ignore")
    merged_df = merged_df.merge(
        text_df[["video_id", "text_ratio"]].dropna(subset=["text_ratio"]),
        on="video_id", how="left"
    )
    return merged_df


def main():
    print(f"[로드] {MERGED.name}")
    merged = pd.read_csv(MERGED, low_memory=False)
    print(f"  현재 컬럼: {list(merged.columns)}")

    # 1. expression 완료 대기 후 병합
    expr_df = wait_for_expression(check_interval=60)
    merged = merge_expression(merged, expr_df)
    print(f"[병합] expression 완료 → non-null: {merged['expression_dominant'].notna().sum():,}")

    # 2. text_ratio 현재까지 분량 병합 (있으면)
    merged = merge_text_ratio(merged)

    # 3. text_region_color 병합 (있으면)
    merged = merge_text_region_color(merged)

    merged.to_csv(MERGED, index=False)
    print(f"\n[저장 완료] {MERGED}")
    print(f"최종 컬럼 ({len(merged.columns)}개): {list(merged.columns)}")
    print(f"\n피처 non-null 현황:")
    feat_cols = [c for c in merged.columns if c not in
                 ["video_id","subcategory","age_~17","age_18~24","age_25~34",
                  "age_35~44","age_45~54","age_55~64","age_65~"]]
    for c in feat_cols:
        print(f"  {c}: {merged[c].notna().sum():,}")


def quick_merge():
    """체크포인트만 merged CSV에 반영 (대기 없음)."""
    print(f"[로드] {MERGED.name}")
    merged = pd.read_csv(MERGED, low_memory=False)
    merged = merge_text_ratio(merged)
    merged = merge_text_region_color(merged)
    merged.to_csv(MERGED, index=False)
    print(f"[저장 완료] {MERGED}")
    for col in TEXT_COLOR_COLS:
        if col in merged.columns:
            print(f"  {col}: {merged[col].notna().sum():,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="체크포인트만 병합 (대기 없음)")
    args = parser.parse_args()
    if args.quick:
        quick_merge()
    else:
        main()
