import matplotlib.pyplot as plt
import numpy as np
import lovelyplots
import os

# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Set matplotlib style
plt.style.use(['ipynb', 'use_mathtext', 'colors5-light'])
plt.rc('font', family='NanumGothic')

# 1. Visualize price elasticity comparison between BEV and ICE
plt.figure(figsize=(12, 8), dpi=150)

# Create bar positions
segments = elasticity_df.index
x = np.arange(len(segments))
width = 0.35

# Create bars
plt.bar(x - width/2, elasticity_df['bev_elasticity'], width, label='전기차 (BEV)', color='skyblue')
plt.bar(x + width/2, elasticity_df['ice_elasticity'], width, label='내연기관차 (ICE)', color='lightcoral')

# Add horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Add horizontal lines for elasticity thresholds
plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
plt.text(len(segments)-1, -1.1, '탄력적/비탄력적 경계', ha='right', fontsize=10)

# Add labels and title
plt.xlabel('차량 세그먼트', fontsize=14)
plt.ylabel('가격 탄력성', fontsize=14)
plt.title('세그먼트별 전기차와 내연기관차의 가격 탄력성 비교', fontsize=16)
plt.xticks(x, segments, fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add market share as text on top of the chart
for i, segment in enumerate(segments):
    plt.text(i, -13, f"시장점유율: {elasticity_df.loc[segment, 'market_share']}%", 
             ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('./artifacts/price_elasticity_comparison.png')
plt.close()

# 2. Create a scatter plot of market share vs elasticity
plt.figure(figsize=(10, 8), dpi=150)

# BEV elasticity vs market share
plt.scatter(elasticity_df['market_share'], elasticity_df['bev_elasticity'], 
           s=100, alpha=0.7, color='blue', label='전기차 (BEV)')

# ICE elasticity vs market share
plt.scatter(elasticity_df['market_share'], elasticity_df['ice_elasticity'], 
           s=100, alpha=0.7, color='red', label='내연기관차 (ICE)')

# Add segment labels to each point
for i, segment in enumerate(elasticity_df.index):
    plt.annotate(segment, 
                (elasticity_df.loc[segment, 'market_share'], elasticity_df.loc[segment, 'bev_elasticity']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.annotate(segment, 
                (elasticity_df.loc[segment, 'market_share'], elasticity_df.loc[segment, 'ice_elasticity']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

# Add horizontal line at y=0 and y=-1
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
plt.text(elasticity_df['market_share'].max(), -1.1, '탄력적/비탄력적 경계', ha='right', fontsize=10)

plt.title('시장 점유율과 가격 탄력성의 관계', fontsize=16)
plt.xlabel('시장 점유율 (%)', fontsize=14)
plt.ylabel('가격 탄력성', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./artifacts/market_share_vs_elasticity.png')
plt.close()

print("추가 시각화 파일이 생성되었습니다.")