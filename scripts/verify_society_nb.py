import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('2_data_analyze_pipeline/05_society_backbone_clusterer_selection.ipynb', encoding='utf-8'))
cells = nb['cells']

print(f"총 셀 수: {len(cells)}")
print()

checks = {
    'SOCIETY CSV 경로': 'SOCIETY_final_kr_clean_add_color_text_person_ratio_title.csv',
    'CATEGORY=SOCIETY': "CATEGORY = 'SOCIETY'",
    'thumbnail_path_x 사용': 'thumbnail_path_x',
    'channel_id_x 사용': 'channel_id_x',
    'channel_name_x 사용': 'channel_name_x',
    'FEATURES_CSV 제거 확인 (없어야 함)': 'FEATURES_CSV',
    '피처 집계 로직': 'groupby',
    'OUT_BASE=SOCIETY': "artifacts' / CATEGORY / 'analysis",
}

full_src = '\n'.join(''.join(c['source']) for c in cells)

for desc, keyword in checks.items():
    found = keyword in full_src
    if desc.startswith('FEATURES_CSV 제거'):
        status = 'PASS' if not found else 'FAIL (아직 남아있음)'
    else:
        status = 'PASS' if found else 'FAIL (없음)'
    print(f"  [{status}] {desc}")

print()
# VLOG 잔재 확인
vlog_remains = []
for kw in ['YT_dataset_VLOG', "CATEGORY = 'VLOG'", 'thumbnails_VLOG', 'channel_id\']', "'channel_name'"]:
    if kw in full_src:
        vlog_remains.append(kw)
if vlog_remains:
    print(f"[warn] VLOG 잔재 키워드: {vlog_remains}")
else:
    print("  [PASS] VLOG 잔재 없음")
