import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "2_data_analyze_pipeline" / "05_society_backbone_clusterer_selection.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

s5  = "".join(nb["cells"][5]["source"])
s13 = "".join(nb["cells"][13]["source"])
s17 = "".join(nb["cells"][17]["source"])

checks = {
    "siglip removed from s5":        "siglip" not in s5,
    "OMP_NUM_THREADS=14":            "'OMP_NUM_THREADS', '14'" in s5,
    "HDBSCAN mcs=[20,30]":           "[20, 30]" in s5,
    "HDBSCAN ms=[None,5]":           "[None, 5]" in s5,
    "tick_labels in cell13 (2x)":    s13.count("tick_labels") == 2,
    "old labels= gone from cell13":  ("boxplot(data_ari, labels=" not in s13) and ("tick_labels" in s13),
    "tick_labels in cell17 (1x)":    s17.count("tick_labels") == 1,
    "old labels= gone from cell17":  ("boxplot(data, labels=" not in s17) and ("tick_labels" in s17),
    "display(summ_df) full":         "display(summ_df)" in s17 and "head(10)" not in s17,
    "임시 comment removed":          "임시" not in s17,
}

all_pass = True
for k, v in checks.items():
    status = "OK" if v else "FAIL"
    if not v:
        all_pass = False
    print(f"  [{status}] {k}")

print()
print("All checks passed!" if all_pass else "Some checks FAILED - review above.")
