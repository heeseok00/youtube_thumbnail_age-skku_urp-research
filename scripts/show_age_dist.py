import pandas as pd

df = pd.read_csv("Data/HEALTH/HEALTH_new_category.csv")

AGE_COLS = ["age_~17","age_18~24","age_25~34","age_35~44","age_45~54","age_55~64","age_65~"]
has_age = df[AGE_COLS].notna().all(axis=1)
d = df[has_age].copy()

d["young"] = d["age_~17"] + d["age_18~24"] + d["age_25~34"]
d["old"]   = d["age_65~"]

print(f"[연령 데이터 있는 영상: {has_age.sum():,}개 / 전체 {len(df):,}개]\n")

header = f"{'subcategory':<12} {'N':>6}  {'~17':>5} {'18~24':>6} {'25~34':>6} {'35~44':>6} {'45~54':>6} {'55~64':>6} {'65~':>5}  {'[~34]':>6} {'[65+]':>6}"
print(header)
print("-" * 88)

order = d.groupby("subcategory").size().sort_values(ascending=False).index
for cat in order:
    grp = d[d["subcategory"] == cat]
    n = len(grp)
    ages = [grp[c].mean() for c in AGE_COLS]
    y = grp["young"].mean()
    o = grp["old"].mean()
    row = f"{str(cat):<12} {n:>6,}  " + "  ".join(f"{a:>5.1f}" for a in ages)
    row += f"  {y:>6.1f} {o:>6.1f}"
    print(row)

print("-" * 88)
ages_all = [d[c].mean() for c in AGE_COLS]
y_all = d["young"].mean()
o_all = d["old"].mean()
row = f"{'TOTAL':<12} {len(d):>6,}  " + "  ".join(f"{a:>5.1f}" for a in ages_all)
row += f"  {y_all:>6.1f} {o_all:>6.1f}"
print(row)
