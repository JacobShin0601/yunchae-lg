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

# 1. Create a comprehensive project classification framework
# Add a column for project size based on capacity
capacity_bins = [0, 10, 30, 50, 100, float('inf')]
capacity_labels = ['Very Small (<10kt)', 'Small (10-30kt)', 'Medium (30-50kt)', 'Large (50-100kt)', 'Very Large (>100kt)']
df['Size_Category'] = pd.cut(df['Capacity'], bins=capacity_bins, labels=capacity_labels)

# Add a column for cost quartiles
df['Cost_Quartile'] = pd.qcut(df['Total_Cash_Cost'], 4, labels=['Q1 (Low Cost)', 'Q2', 'Q3', 'Q4 (High Cost)'])

# Create a comprehensive project classification dataframe
project_classification = df[['project', 'Region', 'country', 'Status(as of 2025.1Q)', 
                           'Extraction Type', 'Resource', 'Geology', 
                           'Total_Cash_Cost', 'Capacity', 'Size_Category', 'Cost_Quartile']]

# Save the comprehensive classification to CSV
project_classification.to_csv('./artifacts/project_classification.csv', index=False)

print("Comprehensive project classification saved to './artifacts/project_classification.csv'")

# 2. Analyze the relationship between project size and cost
size_cost = df.groupby('Size_Category')['Total_Cash_Cost'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
print("\nRelationship between Project Size and Cost:")
print(size_cost)

# Create a visualization of project size vs cost
plt.style.use(['ipynb', 'use_mathtext','colors5-light'])
plt.rc('font', family='NanumGothic')
plt.figure(figsize=(12, 8), dpi=150)

# Create boxplot of Total Cash Cost by Size Category
boxplot = df.boxplot(column='Total_Cash_Cost', by='Size_Category', figsize=(12, 8), 
                    grid=True, rot=45, fontsize=12, return_type='dict')

plt.title('프로젝트 규모별 한계원가 분포', fontsize=16)
plt.suptitle('')  # Remove default suptitle
plt.xlabel('프로젝트 규모', fontsize=14)
plt.ylabel('한계원가 (USD/t)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/cost_by_project_size.png')

# 3. Analyze the relationship between geology and cost
geology_cost = df.groupby('Geology')['Total_Cash_Cost'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
print("\nRelationship between Geology and Cost:")
print(geology_cost)

# Create a visualization of geology vs cost
plt.figure(figsize=(12, 8), dpi=150)

# Create boxplot of Total Cash Cost by Geology
# Filter to include only geologies with at least 5 projects
geology_counts = df['Geology'].value_counts()
common_geologies = geology_counts[geology_counts >= 5].index
df_filtered = df[df['Geology'].isin(common_geologies)]

boxplot = df_filtered.boxplot(column='Total_Cash_Cost', by='Geology', figsize=(12, 8), 
                             grid=True, rot=45, fontsize=12, return_type='dict')

plt.title('지질학적 특성별 한계원가 분포', fontsize=16)
plt.suptitle('')  # Remove default suptitle
plt.xlabel('지질학적 특성', fontsize=14)
plt.ylabel('한계원가 (USD/t)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/cost_by_geology.png')

# 4. Create a matrix of extraction type vs resource type with average cost
extraction_resource_cost = df.pivot_table(
    values='Total_Cash_Cost', 
    index='Extraction Type', 
    columns='Resource', 
    aggfunc='mean'
).round(2)

print("\nAverage Cost by Extraction Type and Resource Type:")
print(extraction_resource_cost)

# Save the matrix to CSV
extraction_resource_cost.to_csv('./artifacts/extraction_resource_cost.csv')

# Create a heatmap visualization
plt.figure(figsize=(12, 8), dpi=150)

# Filter to include only combinations with data
extraction_resource_cost_filtered = extraction_resource_cost.dropna(how='all').dropna(axis=1, how='all')

# Create heatmap
heatmap = plt.imshow(extraction_resource_cost_filtered, cmap='YlOrRd')
plt.colorbar(heatmap, label='평균 한계원가 (USD/t)')
plt.title('추출 방법 및 자원 유형별 평균 한계원가', fontsize=16)
plt.xticks(np.arange(len(extraction_resource_cost_filtered.columns)), 
          extraction_resource_cost_filtered.columns, rotation=45, ha='right')
plt.yticks(np.arange(len(extraction_resource_cost_filtered.index)), 
          extraction_resource_cost_filtered.index)

# Add text annotations to the heatmap
for i in range(len(extraction_resource_cost_filtered.index)):
    for j in range(len(extraction_resource_cost_filtered.columns)):
        if not np.isnan(extraction_resource_cost_filtered.iloc[i, j]):
            plt.text(j, i, f"{extraction_resource_cost_filtered.iloc[i, j]:.0f}", 
                    ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('./artifacts/extraction_resource_cost_heatmap.png')

# 5. Create a framework for analyzing project status transitions
# Group projects by status and analyze their characteristics
status_analysis = df.groupby('Status(as of 2025.1Q)').agg({
    'Total_Cash_Cost': ['mean', 'median', 'min', 'max', 'std', 'count'],
    'Capacity': ['sum', 'mean', 'count']
}).round(2)

print("\nProject Status Analysis:")
print(status_analysis)

# Save the status analysis to CSV
status_analysis.to_csv('./artifacts/status_analysis.csv')

# Create a visualization of average cost by project status
plt.figure(figsize=(12, 6), dpi=150)

# Calculate mean cash cost by status
status_cost = df.groupby('Status(as of 2025.1Q)')['Total_Cash_Cost'].mean().sort_values()

status_cost.plot(kind='barh')
plt.title('프로젝트 상태별 평균 한계원가', fontsize=16)
plt.xlabel('평균 한계원가 (USD/t)', fontsize=14)
plt.ylabel('프로젝트 상태', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/avg_cost_by_status.png')

# 6. Create a country-level analysis framework
# Group projects by country and analyze their characteristics
country_analysis = df.groupby('country').agg({
    'Total_Cash_Cost': ['mean', 'median', 'min', 'max', 'std', 'count'],
    'Capacity': ['sum', 'mean', 'count']
}).round(2)

# Filter to include only countries with at least 3 projects
country_analysis = country_analysis[country_analysis[('Total_Cash_Cost', 'count')] >= 3]

# Sort by average cost
country_analysis = country_analysis.sort_values(('Total_Cash_Cost', 'mean'))

print("\nCountry-level Analysis (countries with at least 3 projects):")
print(country_analysis)

# Save the country analysis to CSV
country_analysis.to_csv('./artifacts/country_analysis.csv')

# Create a visualization of average cost by country
plt.figure(figsize=(14, 10), dpi=150)

# Calculate mean cash cost by country (for countries with at least 3 projects)
countries_with_projects = df['country'].value_counts()[df['country'].value_counts() >= 3].index
df_filtered = df[df['country'].isin(countries_with_projects)]
country_cost = df_filtered.groupby('country')['Total_Cash_Cost'].mean().sort_values()

country_cost.plot(kind='barh')
plt.title('국가별 평균 한계원가 (3개 이상 프로젝트 보유 국가)', fontsize=16)
plt.xlabel('평균 한계원가 (USD/t)', fontsize=14)
plt.ylabel('국가', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/avg_cost_by_country.png')

# 7. Create a framework for analyzing the relationship between project start date and cost
# Convert P.Start to year
df['Start_Year'] = df['P.Start'].dt.year

# Group projects by start year and analyze their characteristics
year_analysis = df.groupby('Start_Year').agg({
    'Total_Cash_Cost': ['mean', 'median', 'min', 'max', 'std', 'count'],
    'Capacity': ['sum', 'mean', 'count']
}).round(2)

# Filter to include only years with at least 3 projects
year_analysis = year_analysis[year_analysis[('Total_Cash_Cost', 'count')] >= 3]

print("\nProject Start Year Analysis (years with at least 3 projects):")
print(year_analysis)

# Save the year analysis to CSV
year_analysis.to_csv('./artifacts/year_analysis.csv')

# Create a visualization of average cost by start year
plt.figure(figsize=(14, 8), dpi=150)

# Calculate mean cash cost by start year (for years with at least 3 projects)
years_with_projects = df['Start_Year'].value_counts()[df['Start_Year'].value_counts() >= 3].index
df_filtered = df[df['Start_Year'].isin(years_with_projects)]
year_cost = df_filtered.groupby('Start_Year')['Total_Cash_Cost'].mean()

year_cost.plot(kind='line', marker='o')
plt.title('프로젝트 시작 연도별 평균 한계원가', fontsize=16)
plt.xlabel('프로젝트 시작 연도', fontsize=14)
plt.ylabel('평균 한계원가 (USD/t)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/avg_cost_by_start_year.png')

# 8. Create a comprehensive data framework for further analysis
# Combine all relevant information into a single dataframe
comprehensive_data = df[['project', 'Region', 'country', 'Status(as of 2025.1Q)', 
                        'Extraction Type', 'Resource', 'Geology', 
                        'Total_Cash_Cost', 'Capacity', 'Size_Category', 'Cost_Quartile', 'Start_Year']]

# Save the comprehensive data to CSV
comprehensive_data.to_csv('./artifacts/comprehensive_project_data.csv', index=False)

print("\nComprehensive project data saved to './artifacts/comprehensive_project_data.csv'")

# 9. Create a stacked bar chart of projects by region and cost quartile
plt.figure(figsize=(14, 8), dpi=150)
region_cost_quartile = pd.crosstab(df['Region'], df['Cost_Quartile'])
region_cost_quartile.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('지역별 비용 사분위 프로젝트 분포', fontsize=16)
plt.xlabel('지역', fontsize=14)
plt.ylabel('프로젝트 수', fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend(title='비용 사분위')
plt.tight_layout()
plt.savefig('./artifacts/region_cost_quartile.png')

# Save the cross-tabulation to CSV
region_cost_quartile.to_csv('./artifacts/region_cost_quartile.csv')

print("\nRegion and cost quartile analysis saved to './artifacts/region_cost_quartile.csv'")
print(region_cost_quartile)

# Result accumulation storage section
# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Result file path
results_file = './artifacts/all_results.txt'
backup_file = './artifacts/all_results_backup_{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# Current analysis parameters
stage_name = "리튬 프로젝트 특성과 한계원가 관계 분석"
result_description = f"""리튬 프로젝트 특성과 한계원가 관계 분석 결과:
- 프로젝트 규모와 한계원가 관계:
{size_cost.to_string()}

- 지질학적 특성과 한계원가 관계:
{geology_cost.to_string()}

- 추출 방법 및 자원 유형별 평균 한계원가:
{extraction_resource_cost.to_string()}

- 프로젝트 상태별 분석:
  - 평균 한계원가가 가장 낮은 상태: {status_cost.index[0]} ({status_cost.iloc[0]:.2f} USD/t)
  - 평균 한계원가가 가장 높은 상태: {status_cost.index[-1]} ({status_cost.iloc[-1]:.2f} USD/t)

- 국가별 분석 (3개 이상 프로젝트 보유 국가):
  - 평균 한계원가가 가장 낮은 국가: {country_cost.index[0]} ({country_cost.iloc[0]:.2f} USD/t)
  - 평균 한계원가가 가장 높은 국가: {country_cost.index[-1]} ({country_cost.iloc[-1]:.2f} USD/t)

- 프로젝트 시작 연도별 분석:
  - 연도별 평균 한계원가 추이 확인
  - 최근 프로젝트일수록 한계원가 변화 패턴 분석

- 지역별 비용 사분위 분포:
{region_cost_quartile.to_string()}
"""

artifact_files = [
    ["./artifacts/project_classification.csv", "프로젝트 분류 프레임워크"],
    ["./artifacts/cost_by_project_size.png", "프로젝트 규모별 한계원가 분포 시각화"],
    ["./artifacts/cost_by_geology.png", "지질학적 특성별 한계원가 분포 시각화"],
    ["./artifacts/extraction_resource_cost.csv", "추출 방법 및 자원 유형별 평균 한계원가 데이터"],
    ["./artifacts/extraction_resource_cost_heatmap.png", "추출 방법 및 자원 유형별 평균 한계원가 히트맵"],
    ["./artifacts/status_analysis.csv", "프로젝트 상태별 분석 데이터"],
    ["./artifacts/avg_cost_by_status.png", "프로젝트 상태별 평균 한계원가 시각화"],
    ["./artifacts/country_analysis.csv", "국가별 분석 데이터"],
    ["./artifacts/avg_cost_by_country.png", "국가별 평균 한계원가 시각화"],
    ["./artifacts/year_analysis.csv", "프로젝트 시작 연도별 분석 데이터"],
    ["./artifacts/avg_cost_by_start_year.png", "프로젝트 시작 연도별 평균 한계원가 시각화"],
    ["./artifacts/comprehensive_project_data.csv", "종합적인 프로젝트 데이터"],
    ["./artifacts/region_cost_quartile.png", "지역별 비용 사분위 프로젝트 분포 시각화"],
    ["./artifacts/region_cost_quartile.csv", "지역별 비용 사분위 프로젝트 분포 데이터"]
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