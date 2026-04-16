import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('2_data_analyze_pipeline/05_society_backbone_clusterer_selection.ipynb', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    for kw in ["channel_id']", "'channel_name'"]:
        if kw in src:
            # 해당 라인만 출력
            for ln, line in enumerate(src.splitlines()):
                if kw in line:
                    print(f"  Cell[{i:02d}] line {ln}: {line.strip()}")
