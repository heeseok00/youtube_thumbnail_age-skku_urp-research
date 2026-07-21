import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")
matplotlib.rcParams["font.family"] = "NanumGothic"

rows = [
    ["Visual",   "text_ratio (MDI=0.393)",      "0.234",              "0.345",               "MW p=2.02e-118"],
    ["Visual",   "color_saturation",             "77.8",               "92.4",                "MW p=8.90e-59"],
    ["Visual",   "color_brightness",             "124.6",              "118.4",               "MW p=6.38e-8"],
    ["Visual",   "color_entropy",                "2.327",              "2.515",               "MW p=2.39e-25"],
    ["Visual",   "expression_dominant",          "positive/neutral",   "negative",            "chi2=66.99"],
    ["Text",     "Attention keywords (Rollout)", "reason·method·most", "secret·heart·moment", "Attention Rollout"],
    ["Caption",  "SHAP tokens",                  "painting·sitting·mask", "suit·crying·yelling", "LLaVA SHAP"],
]

headers = ["Modality", "Feature / Signal", "~34 group", "65+ group", "Evidence"]

col_widths = [0.11, 0.27, 0.20, 0.22, 0.20]
fig_width = 13
row_height = 0.48
header_height = 0.56
n_rows = len(rows)
caption_h = 0.75
fig_height = header_height + n_rows * row_height + 0.1

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_height)
ax.axis("off")

BLACK  = "black"
WHITE  = "white"
LIGHT_GRAY = "#f2f2f2"
BORDER = "black"

xs = [0]
for w in col_widths[:-1]:
    xs.append(xs[-1] + w)

def draw_cell(ax, x, y, w, h, text, bg, fg="black", fontsize=9.5,
              bold=False, align="center", pad=0.010):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=BORDER, linewidth=0.7)
    ax.add_patch(rect)
    if align == "center":
        tx, ha = x + w / 2, "center"
    else:
        tx, ha = x + pad, "left"
    ty = y + h / 2
    ax.text(tx, ty, text, ha=ha, va="center", fontsize=fontsize, color=fg,
            fontweight="bold" if bold else "normal")

# ── Header row ────────────────────────────────────────────────────────────────
y_header = fig_height - header_height - 0.05
for i, (hdr, w) in enumerate(zip(headers, col_widths)):
    draw_cell(ax, xs[i], y_header, w, header_height, hdr,
              bg=WHITE, fg=BLACK, fontsize=10, bold=True)

# ── Data rows ─────────────────────────────────────────────────────────────────
prev_modality = None
for r_idx, row in enumerate(rows):
    modality = row[0]
    y = y_header - (r_idx + 1) * row_height
    row_bg = LIGHT_GRAY if r_idx % 2 == 0 else WHITE

    for c_idx, (cell, w) in enumerate(zip(row, col_widths)):
        if c_idx == 0:
            label = modality if modality != prev_modality else ""
            draw_cell(ax, xs[c_idx], y, w, row_height, label,
                      bg=row_bg, bold=(label != ""), align="center")
        else:
            draw_cell(ax, xs[c_idx], y, w, row_height, cell,
                      bg=row_bg, align="left" if c_idx == 1 else "center")
    prev_modality = modality


plt.tight_layout(pad=0.2)
out_path = "/home/urp_jwl2/26-1_URP/table1_discriminating_signals.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
