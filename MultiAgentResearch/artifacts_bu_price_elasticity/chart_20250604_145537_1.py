import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lovelyplots
import os

# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Set matplotlib style
plt.style.use(['ipynb', 'use_mathtext', 'colors5-light'])
plt.rc('font', family='NanumGothic')

# Load price_diff.json
with open('artifacts/price_diff.json', 'r') as f:
    price_diff_data = json.load(f)

# Create a function to extract and process sales data
def extract_sales_data(segment_data):
    bev_sales = segment_data['PRICE_DIFF']['bev_sales']
    ice_sales = segment_data['PRICE_DIFF']['ice_sales']
    
    # Convert to DataFrame
    bev_df = pd.DataFrame.from_dict(bev_sales, orient='index', columns=['sales'])
    bev_df.index = pd.to_datetime(bev_df.index)
    bev_df = bev_df.sort_index().dropna()
    
    ice_df = pd.DataFrame.from_dict(ice_sales, orient='index', columns=['sales'])
    ice_df.index = pd.to_datetime(ice_df.index)
    ice_df = ice_df.sort_index().dropna()
    
    return bev_df, ice_df

# Analyze sales trends for each segment
segments = list(price_diff_data.keys())
plt.figure(figsize=(14, 10), dpi=150)

# Create subplots for each segment
fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=150)
axes = axes.flatten()

for i, segment in enumerate(segments):
    if i < len(axes):
        ax = axes[i]
        bev_sales, ice_sales = extract_sales_data(price_diff_data[segment])
        
        # Plot BEV and ICE sales
        ax.plot(bev_sales.index, bev_sales['sales'], marker='o', label='전기차 (BEV)', color='blue')
        ax.plot(ice_sales.index, ice_sales['sales'], marker='s', label='내연기관차 (ICE)', color='red')
        
        # Add labels and title
        ax.set_title(f'{segment} 세그먼트 판매량 추이', fontsize=14)
        ax.set_xlabel('연도', fontsize=12)
        ax.set_ylabel('판매량', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('./artifacts/sales_trends_by_segment.png')
plt.close()

# Calculate market penetration (BEV sales as percentage of total sales)
plt.figure(figsize=(12, 8), dpi=150)

for segment in segments:
    bev_sales, ice_sales = extract_sales_data(price_diff_data[segment])
    
    # Align the indices
    common_index = bev_sales.index.intersection(ice_sales.index)
    if len(common_index) > 0:
        bev_aligned = bev_sales.loc[common_index]
        ice_aligned = ice_sales.loc[common_index]
        
        # Calculate BEV market share
        total_sales = bev_aligned['sales'] + ice_aligned['sales']
        bev_share = (bev_aligned['sales'] / total_sales) * 100
        
        # Plot BEV market share
        plt.plot(common_index, bev_share, marker='o', label=segment)

plt.title('세그먼트별 전기차 시장 점유율 추이', fontsize=16)
plt.xlabel('연도', fontsize=14)
plt.ylabel('전기차 시장 점유율 (%)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./artifacts/bev_market_share_trends.png')
plt.close()

print("판매량 및 시장 점유율 시각화 파일이 생성되었습니다.")