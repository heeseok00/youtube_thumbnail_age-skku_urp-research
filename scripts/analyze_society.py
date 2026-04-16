import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data/SOCIETY/SOCIETY_final_kr_clean_add_color_text_person_ratio_title.csv', low_memory=False)

print("=== ending_style ===")
print(df['ending_style'].value_counts().to_string())
print()

print("=== age distribution (mean %) ===")
age_cols = ['age_~17','age_18~24','age_25~34','age_35~44','age_45~54','age_55~64','age_65~']
print(df[age_cols].mean().round(2).to_string())
print()

print("=== thumbnail color means ===")
color_cols = ['saturation_mean','laba_mean','labb_mean','color_entropy','saturated_ratio',
              'red_ratio','orange_ratio','yellow_ratio','green_ratio','cyan_ratio','blue_ratio','purple_ratio','pink_ratio']
print(df[color_cols].mean().round(4).to_string())
print()

print("=== channel video count stats ===")
ch_cnt = df['channel_name_x'].value_counts()
print(f"total channels: {ch_cnt.shape[0]:,}")
print(f"mean: {ch_cnt.mean():.1f}, median: {ch_cnt.median():.0f}, max: {ch_cnt.max()}")
print()

print("=== view_count quantiles ===")
print(df['view_count_x'].quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).round(0).to_string())
print()

print("=== subscriber tier distribution ===")
bins = [0,1000,10000,100000,1000000,float('inf')]
labels = ['~1K','1K~10K','10K~100K','100K~1M','1M~']
df['sub_tier'] = pd.cut(df['subscriberCount'], bins=bins, labels=labels)
print(df['sub_tier'].value_counts().sort_index().to_string())
print()

print("=== text_ratio & person_ratio ===")
print(df[['text_ratio','person_ratio']].describe().round(3).to_string())
print()

print("=== top 10 channels by video count ===")
print(ch_cnt.head(10).to_string())
