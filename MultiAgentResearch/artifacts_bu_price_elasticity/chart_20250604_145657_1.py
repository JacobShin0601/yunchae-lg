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

# Create a function to extract Granger causality data
def extract_granger_data(segment_data):
    granger = segment_data['GRANGER CAUSALITY(MONTH)']
    
    # Convert to DataFrame
    lags = [int(lag.split()[1]) for lag in granger.keys()]
    p_values = list(granger.values())
    
    granger_df = pd.DataFrame({
        'lag': lags,
        'p_value': p_values
    })
    
    return granger_df

# Create a heatmap of Granger causality p-values
segments = list(price_diff_data.keys())
max_lag = 12  # Assuming all segments have the same number of lags

# Create a DataFrame to hold p-values for all segments
granger_matrix = pd.DataFrame(index=segments, columns=range(1, max_lag+1))

for segment in segments:
    granger_df = extract_granger_data(price_diff_data[segment])
    for _, row in granger_df.iterrows():
        granger_matrix.loc[segment, row['lag']] = row['p_value']

# Create heatmap
plt.figure(figsize=(14, 8), dpi=150)
im = plt.imshow(granger_matrix.values, cmap='YlOrRd_r', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('p-value', fontsize=12)

# Add significance threshold line in the colorbar
cbar.ax.axhline(y=0.05, color='black', linestyle='--')
cbar.ax.text(0.5, 0.05, '0.05', ha='center', va='center', color='black', fontsize=10)

# Add labels and title
plt.title('세그먼트별 Granger 인과성 검정 p-값 (가격차이 → 판매량)', fontsize=16)
plt.xlabel('시차 (월)', fontsize=14)
plt.ylabel('차량 세그먼트', fontsize=14)

# Set x and y ticks
plt.xticks(np.arange(max_lag), np.arange(1, max_lag+1))
plt.yticks(np.arange(len(segments)), segments)

# Add grid
for i in range(len(segments)):
    for j in range(max_lag):
        p_value = granger_matrix.iloc[i, j]
        if not pd.isna(p_value):
            color = 'white' if p_value < 0.05 else 'black'
            plt.text(j, i, f'{p_value:.3f}', ha='center', va='center', color=color, fontsize=8)
            if p_value < 0.05:
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=2))

plt.tight_layout()
plt.savefig('./artifacts/granger_causality_heatmap.png')
plt.close()

# Create a summary of significant Granger causality results
significant_lags = {}
for segment in segments:
    granger_df = extract_granger_data(price_diff_data[segment])
    sig_lags = granger_df[granger_df['p_value'] < 0.05]['lag'].tolist()
    if sig_lags:
        significant_lags[segment] = sig_lags

print("=== 유의미한 Granger 인과성 결과 (p < 0.05) ===")
for segment, lags in significant_lags.items():
    print(f"{segment}: 시차 {', '.join(map(str, lags))}개월")

# Create a bar chart showing the number of significant lags per segment
plt.figure(figsize=(10, 6), dpi=150)
segments_with_sig = list(significant_lags.keys())
num_sig_lags = [len(significant_lags[segment]) for segment in segments_with_sig]

if segments_with_sig:
    plt.bar(segments_with_sig, num_sig_lags, color='lightgreen')
    plt.title('세그먼트별 유의미한 Granger 인과성 시차 수 (p < 0.05)', fontsize=16)
    plt.xlabel('차량 세그먼트', fontsize=14)
    plt.ylabel('유의미한 시차 수', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./artifacts/significant_granger_lags_bar.png')
plt.close()

print("Granger 인과성 분석 시각화 파일이 생성되었습니다.")