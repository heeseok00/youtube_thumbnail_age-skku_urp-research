"""
Refactoring script for 05_society_backbone_clusterer_selection.ipynb
Applies 6 changes from the notebook_refactoring plan.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "2_data_analyze_pipeline" / "05_society_backbone_clusterer_selection.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))


def src(cell):
    return "".join(cell.get("source", []))


def set_src(cell, code: str):
    lines = code.split("\n")
    cell["source"] = [l + "\n" for l in lines]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")


# ── cell 5: global settings ───────────────────────────────────────────────
s5 = src(nb["cells"][5])

# 1. Remove siglip2-base from BACKBONES
s5 = s5.replace(
    "BACKBONES = {\n    'dinov2-base':  'facebook/dinov2-base',\n    'siglip2-base': 'google/siglip2-base-patch16-224',\n}",
    "BACKBONES = {\n    'dinov2-base': 'facebook/dinov2-base',\n}",
)

# 2. OMP_NUM_THREADS 4 -> 14
s5 = s5.replace(
    "os.environ.setdefault('OMP_NUM_THREADS', '4')",
    "os.environ.setdefault('OMP_NUM_THREADS', '14')",
)

# 3. HDBSCAN sweep range
s5 = s5.replace(
    "HDBSCAN_MIN_CLUSTER_SIZE_LIST = [12, 16, 20, 30, 40]",
    "HDBSCAN_MIN_CLUSTER_SIZE_LIST = [20, 30]",
)
s5 = s5.replace(
    "HDBSCAN_MIN_SAMPLES_LIST   = [None, 5, 8, 10]",
    "HDBSCAN_MIN_SAMPLES_LIST   = [None, 5]",
)

set_src(nb["cells"][5], s5)
print("cell5: siglip removed, OMP_NUM_THREADS=14, HDBSCAN range reduced")

# ── cell 13: seed/subsample stability — Plot 5 boxplots ─────────────────
s13 = src(nb["cells"][13])
s13 = s13.replace(
    "axes[0].boxplot(data_ari, labels=[r.split('|', 1)[1] for r in run_order], showfliers=False)",
    "axes[0].boxplot(data_ari, tick_labels=[r.split('|', 1)[1] for r in run_order], showfliers=False)",
)
s13 = s13.replace(
    "axes[1].boxplot(data_nmi, labels=[r.split('|', 1)[1] for r in run_order], showfliers=False)",
    "axes[1].boxplot(data_nmi, tick_labels=[r.split('|', 1)[1] for r in run_order], showfliers=False)",
)
set_src(nb["cells"][13], s13)
print(f"cell13: tick_labels replaced, count={s13.count('tick_labels')}")

# ── cell 17: feature interpretation ──────────────────────────────────────
s17 = src(nb["cells"][17])

# 4. boxplot tick_labels (Plot 9)
s17 = s17.replace(
    "ax.boxplot(data, labels=[str(k) for k in cls], showfliers=False)",
    "ax.boxplot(data, tick_labels=[str(k) for k in cls], showfliers=False)",
)

# 5. Full summ_df display
s17 = s17.replace("display(summ_df.head(10))", "display(summ_df)")

# 6. Remove '임시' from comment
s17 = s17.replace(
    "# 최종 후보(임시): valid_df 1위",
    "# 최종 후보: valid_df score 1위 자동 선택",
)

set_src(nb["cells"][17], s17)
print(
    f"cell17: tick_labels={s17.count('tick_labels')}, "
    f"display(summ_df)={'display(summ_df)' in s17}, "
    f"comment_fixed={'임시' not in s17}"
)

# ── save ──────────────────────────────────────────────────────────────────
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("notebook saved:", NB_PATH)
