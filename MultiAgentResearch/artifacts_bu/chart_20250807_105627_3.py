import pandas as pd
import matplotlib.pyplot as plt
import lovelyplots
import numpy as np
import os
from datetime import datetime

# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Define file path
file_path = 'data/LithiumProduction/Project_list_250611.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# 1. Analyze Total Cash Cost by Extraction Type
plt.style.use(['ipynb', 'use_mathtext','colors5-light'])
plt.rc('font', family='NanumGothic')
plt.figure(figsize=(14, 8), dpi=150)

# Create a boxplot of Total Cash Cost by Extraction Type
extraction_types = df['Extraction Type'].value_counts().index[:5]  # Top 5 extraction types
df_filtered = df[df['Extraction Type'].isin(extraction_types)]

boxplot = df_filtered.boxplot(column='Total_Cash_Cost', by='Extraction Type', figsize=(14, 8), 
                             grid=True, rot=45, fontsize=12, return_type='dict')

plt.title('추출 방법별 한계원가 분포', fontsize=16)
plt.suptitle('')  # Remove default suptitle
plt.xlabel('추출 방법', fontsize=14)
plt.ylabel('한계원가 (USD/t)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/cost_by_extraction_type.png')

# 2. Analyze Total Cash Cost by Region and Status
plt.figure(figsize=(16, 10), dpi=150)

# Create a grouped boxplot of Total Cash Cost by Region and Status
regions = df['Region'].value_counts().index[:5]  # Top 5 regions
statuses = ['Existing operations', 'Committed projects', 'Probable projects']
df_filtered = df[(df['Region'].isin(regions)) & (df['Status(as of 2025.1Q)'].isin(statuses))]

# Create pivot table for heatmap
pivot_table = df_filtered.pivot_table(
    values='Total_Cash_Cost', 
    index='Region', 
    columns='Status(as of 2025.1Q)', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 8), dpi=150)
heatmap = plt.imshow(pivot_table, cmap='YlOrRd')
plt.colorbar(heatmap, label='평균 한계원가 (USD/t)')
plt.title('지역 및 프로젝트 상태별 평균 한계원가', fontsize=16)
plt.xticks(np.arange(len(pivot_table.columns)), pivot_table.columns, rotation=45, ha='right')
plt.yticks(np.arange(len(pivot_table.index)), pivot_table.index)

# Add text annotations to the heatmap
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        if not np.isnan(pivot_table.iloc[i, j]):
            plt.text(j, i, f"{pivot_table.iloc[i, j]:.0f}", 
                    ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('./artifacts/cost_by_region_status_heatmap.png')

# 3. Create a scatter plot of Capacity vs Total Cash Cost
plt.figure(figsize=(12, 8), dpi=150)

# Filter out extreme outliers for better visualization
q1 = df['Total_Cash_Cost'].quantile(0.01)
q3 = df['Total_Cash_Cost'].quantile(0.99)
df_filtered = df[(df['Total_Cash_Cost'] >= q1) & (df['Total_Cash_Cost'] <= q3)]

# Create scatter plot with color by region
regions = df_filtered['Region'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))

for i, region in enumerate(regions):
    region_data = df_filtered[df_filtered['Region'] == region]
    plt.scatter(region_data['Capacity'], region_data['Total_Cash_Cost'], 
               label=region, color=colors[i], alpha=0.7)

plt.title('생산능력과 한계원가의 관계', fontsize=16)
plt.xlabel('생산능력 (kt/y)', fontsize=14)
plt.ylabel('한계원가 (USD/t)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(title='지역')
plt.tight_layout()
plt.savefig('./artifacts/capacity_vs_cost_scatter.png')

# 4. Create a dataframe with project classification by cost quartiles
df['Cost_Quartile'] = pd.qcut(df['Total_Cash_Cost'], 4, labels=['Q1 (Low Cost)', 'Q2', 'Q3', 'Q4 (High Cost)'])

# Count projects by region and cost quartile
region_cost_quartile = pd.crosstab(df['Region'], df['Cost_Quartile'])
print("Projects by Region and Cost Quartile:")
print(region_cost_quartile)

# Save the cross-tabulation to CSV
region_cost_quartile.to_csv('./artifacts/region_cost_quartile.csv')

# 5. Create a stacked bar chart of projects by region and cost quartile
plt.figure(figsize=(14, 8), dpi=150)
region_cost_quartile.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('지역별 비용 사분위 프로젝트 분포', fontsize=16)
plt.xlabel('지역', fontsize=14)
plt.ylabel('프로젝트 수', fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend(title='비용 사분위')
plt.tight_layout()
plt.savefig('./artifacts/region_cost_quartile.png')

# 6. Create a dataframe with project details for further analysis
project_details = df[['project', 'Region', 'country', 'Status(as of 2025.1Q)', 
                     'Extraction Type', 'Resource', 'Total_Cash_Cost', 'Capacity', 'Cost_Quartile']]

# Sort by Total Cash Cost
project_details_sorted = project_details.sort_values('Total_Cash_Cost')

# Save to CSV for further analysis
project_details_sorted.to_csv('./artifacts/project_details_by_cost.csv', index=False)

print("\nProject details sorted by cost saved to './artifacts/project_details_by_cost.csv'")

# 7. Analyze the lowest cost projects (bottom 10%)
low_cost_threshold = df['Total_Cash_Cost'].quantile(0.1)
low_cost_projects = df[df['Total_Cash_Cost'] <= low_cost_threshold]

print("\nLowest Cost Projects (Bottom 10%):")
print(f"Number of projects: {len(low_cost_projects)}")
print("Distribution by region:")
print(low_cost_projects['Region'].value_counts())
print("\nDistribution by extraction type:")
print(low_cost_projects['Extraction Type'].value_counts())
print("\nDistribution by resource type:")
print(low_cost_projects['Resource'].value_counts())

# 8. Analyze the highest cost projects (top 10%)
high_cost_threshold = df['Total_Cash_Cost'].quantile(0.9)
high_cost_projects = df[df['Total_Cash_Cost'] >= high_cost_threshold]

print("\nHighest Cost Projects (Top 10%):")
print(f"Number of projects: {len(high_cost_projects)}")
print("Distribution by region:")
print(high_cost_projects['Region'].value_counts())
print("\nDistribution by extraction type:")
print(high_cost_projects['Extraction Type'].value_counts())
print("\nDistribution by resource type:")
print(high_cost_projects['Resource'].value_counts())

# 9. Create a framework for regional cost analysis
regional_cost_analysis = df.groupby('Region').agg({
    'Total_Cash_Cost': ['mean', 'median', 'min', 'max', 'std', 'count'],
    'Capacity': ['sum', 'mean', 'count']
}).round(2)

print("\nRegional Cost Analysis Framework:")
print(regional_cost_analysis)

# Save the regional cost analysis to CSV
regional_cost_analysis.to_csv('./artifacts/regional_cost_analysis.csv')

# Result accumulation storage section
# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Result file path
results_file = './artifacts/all_results.txt'
backup_file = './artifacts/all_results_backup_{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# Current analysis parameters
stage_name = "리튬 프로젝트 한계원가 분석"
result_description = f"""리튬 프로젝트 한계원가 분석 결과:
- 추출 방법별 한계원가 분석: 
  - 추출 방법에 따라 한계원가의 차이가 뚜렷함
  - 주요 추출 방법: {', '.join(extraction_types)}

- 지역 및 프로젝트 상태별 한계원가:
  - 지역별 평균 한계원가 차이가 큼
  - 프로젝트 상태에 따라 한계원가 차이 발생

- 비용 사분위별 지역 분포:
{region_cost_quartile.to_string()}

- 저비용 프로젝트(하위 10%) 특성:
  - 프로젝트 수: {len(low_cost_projects)}
  - 지역 분포: {dict(low_cost_projects['Region'].value_counts())}
  - 추출 방법 분포: {dict(low_cost_projects['Extraction Type'].value_counts())}
  - 자원 유형 분포: {dict(low_cost_projects['Resource'].value_counts())}

- 고비용 프로젝트(상위 10%) 특성:
  - 프로젝트 수: {len(high_cost_projects)}
  - 지역 분포: {dict(high_cost_projects['Region'].value_counts())}
  - 추출 방법 분포: {dict(high_cost_projects['Extraction Type'].value_counts())}
  - 자원 유형 분포: {dict(high_cost_projects['Resource'].value_counts())}

- 지역별 한계원가 통계:
  - 평균 한계원가가 가장 낮은 지역: {regional_cost_analysis['Total_Cash_Cost']['mean'].idxmin()} ({regional_cost_analysis['Total_Cash_Cost']['mean'].min():.2f} USD/t)
  - 평균 한계원가가 가장 높은 지역: {regional_cost_analysis['Total_Cash_Cost']['mean'].idxmax()} ({regional_cost_analysis['Total_Cash_Cost']['mean'].max():.2f} USD/t)
"""

artifact_files = [
    ["./artifacts/cost_by_extraction_type.png", "추출 방법별 한계원가 분포 시각화"],
    ["./artifacts/cost_by_region_status_heatmap.png", "지역 및 프로젝트 상태별 평균 한계원가 히트맵"],
    ["./artifacts/capacity_vs_cost_scatter.png", "생산능력과 한계원가의 관계 산점도"],
    ["./artifacts/region_cost_quartile.png", "지역별 비용 사분위 프로젝트 분포 시각화"],
    ["./artifacts/region_cost_quartile.csv", "지역별 비용 사분위 프로젝트 분포 데이터"],
    ["./artifacts/project_details_by_cost.csv", "비용별 정렬된 프로젝트 상세 정보"],
    ["./artifacts/regional_cost_analysis.csv", "지역별 한계원가 분석 프레임워크"]
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