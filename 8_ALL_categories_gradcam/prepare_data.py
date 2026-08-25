"""4카테고리 표본 CSV를 하나로 병합 → outputs/all_sample.csv"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/urp_jwl/URP_backup/26-1_URP")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "category": "SOCIETY",
        "csv": Path("/home/urp_jwl/urp_yena/society_sampled_2600_caption.csv"),
    },
    {
        "category": "EDU",
        "csv": Path(
            "/home/urp_jwl/URP_backup/urp_bin/URP_논문/image captioning/EDU_sampled_2600.csv"
        ),
    },
    {
        "category": "HEALTH",
        "csv": ROOT
        / "7_HEALTH_다이어트_분석/02_image_caption/outputs/diet_sample_2500_vlm.csv",
    },
    {
        "category": "MEDITATION",
        "csv": Path(
            "/home/urp_jwl/urp_jungeun/outputs/image_captioning/meditation_sampled_vlm.csv"
        ),
    },
]

KEEP = [
    "category",
    "video_id",
    "channel_id",
    "channel_name",
    "title",
    "thumbnail_path",
    "resolved_path",
    "y_grouped",
    "target",
]


def resolve_path(p: str) -> Path | None:
    if pd.isna(p) or str(p).strip() == "":
        return None
    s = str(p).strip().replace("\\", "/")
    cand = Path(s)
    if cand.is_absolute() and cand.exists():
        return cand
    cand = ROOT / s
    if cand.exists():
        return cand
    return None


def normalize_age(row: pd.Series) -> tuple[str, int]:
    if "y_grouped" in row.index and pd.notna(row.get("y_grouped")):
        yg = str(row["y_grouped"])
        if yg in ("~34", "34-"):
            return "~34", 0
        if yg in ("65~", "65+"):
            return "65~", 1
    if "age_group" in row.index and pd.notna(row.get("age_group")):
        ag = str(row["age_group"])
        if ag in ("34-", "~34"):
            return "~34", 0
        if ag in ("65+", "65~"):
            return "65~", 1
    if "target" in row.index and pd.notna(row.get("target")):
        t = int(row["target"])
        return ("~34", 0) if t == 0 else ("65~", 1)
    raise ValueError("cannot infer age label")


def load_one(src: dict) -> pd.DataFrame:
    df = pd.read_csv(src["csv"], low_memory=False)
    rows = []
    for _, r in df.iterrows():
        yg, tgt = normalize_age(r)
        path_raw = r.get("resolved_path", r.get("thumbnail_path"))
        rp = resolve_path(path_raw)
        if rp is None:
            # try thumbnail_path if resolved failed
            rp = resolve_path(r.get("thumbnail_path"))
        if rp is None:
            continue
        rows.append(
            {
                "category": src["category"],
                "video_id": r.get("video_id"),
                "channel_id": r.get("channel_id"),
                "channel_name": r.get("channel_name"),
                "title": r.get("title"),
                "thumbnail_path": r.get("thumbnail_path"),
                "resolved_path": str(rp),
                "y_grouped": yg,
                "target": tgt,
            }
        )
    out = pd.DataFrame(rows)
    print(
        f"[{src['category']}] csv={len(df)} kept={len(out)} "
        f"age={out['y_grouped'].value_counts().to_dict()}"
    )
    return out


def main():
    parts = [load_one(s) for s in SOURCES]
    all_df = pd.concat(parts, ignore_index=True)
    # video_id 중복 시 첫 카테고리 유지
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)
    print(f"\nmerged={before} -> unique video_id={len(all_df)}")
    print("by category:", all_df["category"].value_counts().to_dict())
    print("by age:", all_df["y_grouped"].value_counts().to_dict())
    out_path = OUT / "all_sample.csv"
    all_df[KEEP].to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
