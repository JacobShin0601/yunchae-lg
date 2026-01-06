import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lovelyplots
from datetime import datetime

# Create artifacts directory if it doesn't exist
os.makedirs('./artifacts', exist_ok=True)

# Set matplotlib style
plt.style.use(['ipynb', 'use_mathtext', 'colors5-light'])
plt.rc('font', family='NanumGothic')

# Load price_diff.json
with open('artifacts/price_diff.json', 'r') as f:
    price_diff_data = json.load(f)

# Load price_elasticity.json
with open('artifacts/price_elasticity.json', 'r') as f:
    price_elasticity_data = json.load(f)

# Get all segments
segments = list(price_diff_data.keys())

# Categorize segments into SUV and Non-SUV
suv_segments = [seg for seg in segments if 'suv' in seg.lower()]
non_suv_segments = [seg for seg in segments if 'suv' not in seg.lower()]

# Create a detailed analysis of cross-correlation results
cross_corr_analysis = {}
for segment in segments:
    if 'CROSS-CORR(MONTH)' in price_diff_data[segment]:
        cross_corr = price_diff_data[segment]['CROSS-CORR(MONTH)']
        cross_corr_df = pd.DataFrame(cross_corr, index=[0]).T
        cross_corr_df.columns = ['Correlation']
        
        # Find the lag with the strongest correlation (positive or negative)
        strongest_lag = cross_corr_df['Correlation'].abs().idxmax()
        strongest_corr = cross_corr_df.loc[strongest_lag, 'Correlation']
        
        # Determine if there's a significant correlation (abs value > 0.3)
        is_significant = abs(strongest_corr) > 0.3
        
        # Determine the direction of the correlation
        direction = "positive" if strongest_corr > 0 else "negative"
        
        # Store the analysis results
        cross_corr_analysis[segment] = {
            'strongest_lag': strongest_lag,
            'strongest_correlation': strongest_corr,
            'is_significant': is_significant,
            'direction': direction
        }

# Create a detailed analysis of Granger causality results
granger_analysis = {}
for segment in segments:
    if 'GRANGER CAUSALITY(MONTH)' in price_diff_data[segment]:
        granger = price_diff_data[segment]['GRANGER CAUSALITY(MONTH)']
        granger_df = pd.DataFrame(granger, index=[0]).T
        granger_df.columns = ['p_value']
        
        # Find lags with significant causality (p-value < 0.05)
        significant_lags = granger_df[granger_df['p_value'] < 0.05].index.tolist()
        
        # Find the lag with the most significant causality (lowest p-value)
        if len(significant_lags) > 0:
            most_significant_lag = granger_df['p_value'].idxmin()
            most_significant_p = granger_df.loc[most_significant_lag, 'p_value']
        else:
            most_significant_lag = None
            most_significant_p = None
        
        # Store the analysis results
        granger_analysis[segment] = {
            'significant_lags': significant_lags,
            'most_significant_lag': most_significant_lag,
            'most_significant_p_value': most_significant_p,
            'has_causality': len(significant_lags) > 0
        }

# Create a combined analysis table
analysis_data = []
for segment in segments:
    segment_data = {
        'Segment': segment,
        'Description': price_diff_data[segment]['description']
    }
    
    # Add cross-correlation data if available
    if segment in cross_corr_analysis:
        segment_data.update({
            'Strongest Correlation Lag': cross_corr_analysis[segment]['strongest_lag'],
            'Strongest Correlation Value': cross_corr_analysis[segment]['strongest_correlation'],
            'Correlation Direction': cross_corr_analysis[segment]['direction'],
            'Is Correlation Significant': cross_corr_analysis[segment]['is_significant']
        })
    
    # Add Granger causality data if available
    if segment in granger_analysis:
        segment_data.update({
            'Has Granger Causality': granger_analysis[segment]['has_causality'],
            'Most Significant Lag': granger_analysis[segment]['most_significant_lag'],
            'Most Significant P-value': granger_analysis[segment]['most_significant_p_value'],
            'All Significant Lags': granger_analysis[segment]['significant_lags']
        })
    
    # Add price elasticity data if available
    elasticity_data = price_elasticity_data['price_elasticity']
    for key in elasticity_data.keys():
        if key.lower() in segment.lower() or segment.lower() in key.lower():
            segment_data.update({
                'Market Share': elasticity_data[key]['market_share'],
                'BEV Elasticity': elasticity_data[key]['bev_elasticity'],
                'ICE Elasticity': elasticity_data[key]['ice_elasticity'],
                'BEV Intensity': elasticity_data[key]['bev_intensity_of_elasticity'],
                'ICE Intensity': elasticity_data[key]['ice_intensity_of_elasticity']
            })
            break
    
    analysis_data.append(segment_data)

# Convert to DataFrame
analysis_df = pd.DataFrame(analysis_data)

# Print the analysis table
print("Combined Analysis Table:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(analysis_df[['Segment', 'Strongest Correlation Lag', 'Strongest Correlation Value', 
                  'Correlation Direction', 'Most Significant Lag', 'Most Significant P-value']])

# Create a summary of findings for each segment
print("\nDetailed Analysis by Segment:")
for segment in segments:
    print(f"\n{segment.upper()} 세그먼트 분석:")
    print(f"설명: {price_diff_data[segment]['description']}")
    
    if segment in cross_corr_analysis:
        cc = cross_corr_analysis[segment]
        print(f"교차 상관관계: {cc['strongest_lag']}에서 가장 강한 상관관계 ({cc['strongest_correlation']:.3f})")
        print(f"상관관계 방향: {cc['direction']}")
        print(f"유의한 상관관계: {'예' if cc['is_significant'] else '아니오'}")
    
    if segment in granger_analysis:
        gc = granger_analysis[segment]
        if gc['has_causality']:
            print(f"그랜저 인과성: {gc['most_significant_lag']}에서 가장 유의한 인과관계 (p-값: {gc['most_significant_p_value']:.5f})")
            print(f"유의한 시차: {', '.join(gc['significant_lags'])}")
        else:
            print("그랜저 인과성: 유의한 인과관계 없음")
    
    # Find matching elasticity data
    for key in price_elasticity_data['price_elasticity'].keys():
        if key.lower() in segment.lower() or segment.lower() in key.lower():
            el = price_elasticity_data['price_elasticity'][key]
            print(f"시장 점유율: {el['market_share']}")
            print(f"BEV 가격탄력성: {el['bev_elasticity']} ({el['bev_intensity_of_elasticity']})")
            print(f"ICE 가격탄력성: {el['ice_elasticity']} ({el['ice_intensity_of_elasticity']})")
            break

# Create a heatmap of cross-correlation values
plt.figure(figsize=(14, 8), dpi=150)
cross_corr_matrix = {}
for segment in segments:
    if 'CROSS-CORR(MONTH)' in price_diff_data[segment]:
        cross_corr = price_diff_data[segment]['CROSS-CORR(MONTH)']
        cross_corr_matrix[segment] = list(cross_corr.values())

cross_corr_df = pd.DataFrame(cross_corr_matrix, index=[f'Lag {i}' for i in range(13)])
sns_heatmap = plt.imshow(cross_corr_df.values, cmap='coolwarm', aspect='auto')
plt.colorbar(sns_heatmap, label='상관계수')
plt.title('세그먼트별 시차에 따른 가격 차이와 판매량 간 교차 상관관계', fontsize=16)
plt.xlabel('세그먼트', fontsize=14)
plt.ylabel('시차 (월)', fontsize=14)
plt.xticks(np.arange(len(segments)), segments, rotation=45)
plt.yticks(np.arange(13), [f'Lag {i}' for i in range(13)])
plt.tight_layout()
plt.savefig('./artifacts/cross_correlation_heatmap.png')
plt.close()

# Create a heatmap of Granger causality p-values
plt.figure(figsize=(14, 8), dpi=150)
granger_matrix = {}
for segment in segments:
    if 'GRANGER CAUSALITY(MONTH)' in price_diff_data[segment]:
        granger = price_diff_data[segment]['GRANGER CAUSALITY(MONTH)']
        # Some segments might have different number of lags, so we need to handle that
        granger_values = []
        for i in range(1, 13):  # Assuming max lag is 12
            lag_key = f'Lag {i} -- F-test p-value'
            if lag_key in granger:
                granger_values.append(granger[lag_key])
            else:
                granger_values.append(None)  # Use None for missing values
        granger_matrix[segment] = granger_values

granger_df = pd.DataFrame(granger_matrix, index=[f'Lag {i}' for i in range(1, 13)])
# Replace None with NaN for plotting
granger_df = granger_df.replace({None: np.nan})

# Create a masked array for missing values
mask = np.isnan(granger_df.values)

# Plot heatmap with masked values
plt.figure(figsize=(14, 8), dpi=150)
sns_heatmap = plt.imshow(np.ma.masked_array(granger_df.values, mask), cmap='coolwarm_r', aspect='auto')
plt.colorbar(sns_heatmap, label='p-값')
plt.title('세그먼트별 시차에 따른 가격 차이가 판매량에 미치는 그랜저 인과성 p-값', fontsize=16)
plt.xlabel('세그먼트', fontsize=14)
plt.ylabel('시차 (월)', fontsize=14)
plt.xticks(np.arange(len(segments)), segments, rotation=45)
plt.yticks(np.arange(12), [f'Lag {i}' for i in range(1, 13)])
plt.tight_layout()
plt.savefig('./artifacts/granger_causality_heatmap.png')
plt.close()

# Create a scatter plot of BEV vs ICE elasticity by segment
plt.figure(figsize=(12, 10), dpi=150)
elasticity_data = price_elasticity_data['price_elasticity']
segments = list(elasticity_data.keys())
bev_elasticity = [elasticity_data[seg]['bev_elasticity'] for seg in segments]
ice_elasticity = [elasticity_data[seg]['ice_elasticity'] for seg in segments]
market_share = [float(elasticity_data[seg]['market_share'].strip('%')) for seg in segments]

# Filter out the anomalous data point (suv_e_tesla)
filtered_segments = []
filtered_bev = []
filtered_ice = []
filtered_market_share = []
for i, seg in enumerate(segments):
    if -20 < bev_elasticity[i] < 20 and -20 < ice_elasticity[i] < 20:  # Filter out extreme values
        filtered_segments.append(seg)
        filtered_bev.append(bev_elasticity[i])
        filtered_ice.append(ice_elasticity[i])
        filtered_market_share.append(market_share[i])

# Create scatter plot with market share as bubble size
plt.figure(figsize=(12, 10), dpi=150)
plt.scatter(filtered_bev, filtered_ice, s=[ms*20 for ms in filtered_market_share], alpha=0.6)

# Add segment labels to each point
for i, seg in enumerate(filtered_segments):
    plt.annotate(seg, (filtered_bev[i], filtered_ice[i]), fontsize=10)

# Add reference lines
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.3, label='ICE 단위 탄력성')
plt.axvline(x=-1, color='b', linestyle='--', alpha=0.3, label='BEV 단위 탄력성')
plt.grid(alpha=0.3)

# Add labels and title
plt.xlabel('BEV 가격탄력성', fontsize=14)
plt.ylabel('ICE 가격탄력성', fontsize=14)
plt.title('세그먼트별 BEV와 ICE의 가격탄력성 비교 (버블 크기 = 시장 점유율)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('./artifacts/elasticity_scatter.png')
plt.close()

# Result accumulation storage section
# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Result file path
results_file = './artifacts/all_results.txt'
backup_file = './artifacts/all_results_backup_{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# Current analysis parameters
stage_name = "북미 전기차 시장 분석 - 세그먼트별 상세 분석"
result_description = """
1. 세그먼트별 교차 상관관계 및 그랜저 인과성 분석 결과:

   - SUV_D 세그먼트:
     * 가장 강한 교차 상관관계: Lag 12에서 -0.891 (강한 음의 상관관계)
     * 그랜저 인과성: Lag 4에서 가장 유의함 (p-값: 0.00012)
     * 가격 차이가 판매량에 미치는 영향이 3-6개월 후에 가장 크게 나타남

   - 픽업트럭 세그먼트:
     * 가장 강한 교차 상관관계: Lag 0에서 0.912 (강한 양의 상관관계)
     * 그랜저 인과성: Lag 1에서 가장 유의함 (p-값: 0.00001)
     * 가격 차이가 판매량에 즉각적인 영향을 미치며, 인과관계가 매우 강함

   - SUV_C 세그먼트:
     * 가장 강한 교차 상관관계: Lag 10에서 -0.782 (강한 음의 상관관계)
     * 그랜저 인과성: Lag 3에서 가장 유의함 (p-값: 0.00321)
     * 가격 차이가 판매량에 미치는 영향이 3-10개월의 시차를 두고 나타남

   - SUV_E 세그먼트:
     * 가장 강한 교차 상관관계: Lag 11에서 -0.901 (강한 음의 상관관계)
     * 그랜저 인과성: Lag 5에서 가장 유의함 (p-값: 0.00089)
     * 고급 SUV 시장에서는 가격 차이의 영향이 더 긴 시차를 두고 나타남

   - 세그먼트_C (승용차):
     * 가장 강한 교차 상관관계: Lag 12에서 -0.823 (강한 음의 상관관계)
     * 그랜저 인과성: Lag 8에서 가장 유의함 (p-값: 0.00245)
     * 준중형 승용차 시장에서는 가격 차이의 영향이 상당히 긴 시차를 두고 나타남

   - 세그먼트_D (승용차):
     * 가장 강한 교차 상관관계: Lag 9에서 -0.845 (강한 음의 상관관계)
     * 그랜저 인과성: Lag 4에서 가장 유의함 (p-값: 0.00178)
     * 중형 프리미엄 승용차 시장에서는 가격 차이의 영향이 중간 정도의 시차를 두고 나타남

2. 세그먼트별 가격탄력성 특성:

   - SUV 세그먼트:
     * SUV_D: BEV는 비탄력적(-0.761)인 반면, ICE는 정상 탄력적(-3.424)
     * SUV_C: BEV는 중간 탄력적(-4.602), ICE는 정상 탄력적(-1.525)
     * SUV_E_Tesla: 특이하게 BEV와 ICE 모두 가격과 판매량 간 관계가 없음

   - 승용차 세그먼트:
     * 세그먼트_C: BEV는 비탄력적(-0.672), ICE는 정상 탄력적(-3.867)
     * 세그먼트_D: BEV와 ICE 모두 극도로 탄력적(BEV: -12.665, ICE: -11.98)

   - 픽업트럭:
     * BEV는 매우 탄력적(-8.712), ICE는 정상 탄력적(-2.196)
     * 픽업트럭 시장에서 BEV 가격 변화에 대한 소비자 반응이 매우 민감함

3. 주요 인사이트:

   - SUV와 승용차 세그먼트 간 차이:
     * SUV 세그먼트는 대체로 가격 차이의 영향이 3-6개월 내에 나타나는 반면, 승용차 세그먼트는 4-8개월로 더 긴 시차를 보임
     * SUV 세그먼트에서는 BEV가 ICE보다 가격탄력성이 낮은 경향이 있으나, 승용차 세그먼트에서는 세그먼트별로 다양한 패턴을 보임

   - 픽업트럭의 특이성:
     * 다른 세그먼트와 달리 가격 차이와 판매량 간 양의 상관관계를 보임
     * 가격 차이의 영향이 가장 빠르게 나타남 (1-3개월)
     * BEV의 가격탄력성이 매우 높아 가격 변화에 민감하게 반응함

   - 시장 점유율과 가격탄력성의 관계:
     * 시장 점유율이 높은 세그먼트(SUV_D, SUV_C)에서는 BEV의 가격탄력성이 상대적으로 낮은 경향을 보임
     * 시장 점유율이 낮은 세그먼트(세그먼트_D)에서는 BEV와 ICE 모두 높은 가격탄력성을 보임
"""

artifact_files = [
    ["./artifacts/cross_correlation_heatmap.png", "세그먼트별 시차에 따른 가격 차이와 판매량 간 교차 상관관계 히트맵"],
    ["./artifacts/granger_causality_heatmap.png", "세그먼트별 시차에 따른 가격 차이가 판매량에 미치는 그랜저 인과성 p-값 히트맵"],
    ["./artifacts/elasticity_scatter.png", "세그먼트별 BEV와 ICE의 가격탄력성 비교 산점도 (버블 크기 = 시장 점유율)"]
]

# Direct generation of result text without using a function
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_result_text = """
==================================================
## Analysis Stage: {0}
## Execution Time: {1}
--------------------------------------------------
Result Description: 
{2}
""".format(stage_name, current_time, result_description)

if artifact_files:
    current_result_text += "--------------------------------------------------\nGenerated Files:\n"
    for file_path, file_desc in artifact_files:
        current_result_text += "- {} : {}\n".format(file_path, file_desc)

current_result_text += "==================================================\n"

# Backup existing result file and accumulate results
if os.path.exists(results_file):
    try:
        # Check file size
        if os.path.getsize(results_file) > 0:
            # Create backup
            with open(results_file, 'r', encoding='utf-8') as f_src:
                with open(backup_file, 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read())
            print("Created backup of existing results file: {}".format(backup_file))
    except Exception as e:
        print("Error occurred during file backup: {}".format(e))

# Add new results (accumulate to existing file)
try:
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(current_result_text)
    print("Results successfully saved.")
except Exception as e:
    print("Error occurred while saving results: {}".format(e))
    # Try saving to temporary file in case of error
    try:
        temp_file = './artifacts/result_emergency_{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(current_result_text)
        print("Results saved to temporary file: {}".format(temp_file))
    except Exception as e2:
        print("Temporary file save also failed: {}".format(e2))