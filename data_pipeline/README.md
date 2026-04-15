# Data pipeline (stages 00–04)

CSV 구축용 단계별 스크립트·노트북입니다. **실행 시 작업 디렉터리는 저장소 루트**(`Data/`, `.env`, `vling_session.json` 등)를 사용하세요.

| 단계 | 폴더 | 파일 |
|------|------|------|
| 00 | `00_category_selection/` | `00.category_selection.ipynb` |
| 01 | `01_channel_collection/` | `01.channel_collection.ipynb` |
| 02 | `02_youtube_metadata/` | `02.data_collection_metadata.py` |
| 03 | `03_thumbnail_download/` | `03.thumbnail_download.py` |
| 04 | `04_vling_demographics/` | `04.vling_age_sex_extract.py` |

예시 (루트에서):

```powershell
python data_pipeline\02_youtube_metadata\02.data_collection_metadata.py --channels-csv Data\FOOD\FOOD_clean.csv --output-csv Data\FOOD\FOOD_meta.csv --thumbnail-dir Data\FOOD --video-count 10 --skip-thumbnails
python data_pipeline\03_thumbnail_download\03.thumbnail_download.py --meta-csv Data\FOOD\FOOD_meta.csv
python data_pipeline\04_vling_demographics\04.vling_age_sex_extract.py --category SOCIETY
```
