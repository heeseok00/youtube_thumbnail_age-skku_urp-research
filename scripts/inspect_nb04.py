import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('2_data_analyze_pipeline/04_vlog_backbone_clusterer_selection_feature_interpretation.ipynb', encoding='utf-8'))
cells = nb['cells']
print(f"total cells: {len(cells)}")
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    preview = src[:150].replace('\n', ' ')
    print(f"[{i:02d}] {ctype:8s} | {preview}")
