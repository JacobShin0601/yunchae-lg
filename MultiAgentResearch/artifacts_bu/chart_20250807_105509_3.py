import pandas as pd
import matplotlib.pyplot as plt
import lovelyplots
import numpy as np
import os

# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Define file path
file_path = 'data/LithiumProduction/Project_list_250611.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# 1. Analyze projects by region
region_counts = df['Region'].value_counts()
print("Projects by Region:")
print(region_counts)

# 2. Analyze projects by status
status_counts = df['Status(as of 2025.1Q)'].value_counts()
print("\nProjects by Status:")
print(status_counts)

# 3. Analyze projects by resource type
resource_counts = df['Resource'].value_counts()
print("\nProjects by Resource Type:")
print(resource_counts)

# 4. Analyze projects by extraction type
extraction_counts = df['Extraction Type'].value_counts()
print("\nProjects by Extraction Type:")
print(extraction_counts)

# 5. Analyze projects by country
country_counts = df['country'].value_counts().head(10)
print("\nTop 10 Countries by Project Count:")
print(country_counts)

# 6. Analyze Total Cash Cost distribution
print("\nTotal Cash Cost Statistics:")
print(df['Total_Cash_Cost'].describe())

# 7. Create a visualization of projects by region
plt.style.use(['ipynb', 'use_mathtext','colors5-light'])
plt.rc('font', family='NanumGothic')
plt.figure(figsize=(12, 6), dpi=150)

region_counts.plot(kind='bar')
plt.title('리튬 프로젝트 지역별 분포', fontsize=16)
plt.xlabel('지역', fontsize=14)
plt.ylabel('프로젝트 수', fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/projects_by_region.png')

# 8. Create a visualization of Total Cash Cost by Region
plt.figure(figsize=(12, 8), dpi=150)

# Calculate mean cash cost by region
region_cost = df.groupby('Region')['Total_Cash_Cost'].mean().sort_values()

region_cost.plot(kind='barh')
plt.title('지역별 평균 한계원가 (Total Cash Cost)', fontsize=16)
plt.xlabel('평균 한계원가 (USD/t)', fontsize=14)
plt.ylabel('지역', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/avg_cost_by_region.png')

# 9. Create a visualization of projects by status
plt.figure(figsize=(12, 6), dpi=150)

status_counts.plot(kind='bar')
plt.title('리튬 프로젝트 상태별 분포', fontsize=16)
plt.xlabel('프로젝트 상태', fontsize=14)
plt.ylabel('프로젝트 수', fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./artifacts/projects_by_status.png')

# 10. Extract project list and save to CSV
project_list = df[['project', 'Region', 'country', 'Status(as of 2025.1Q)', 'Total_Cash_Cost', 'Capacity']]
project_list.to_csv('./artifacts/lithium_project_list.csv', index=False, encoding='utf-8')

print("\nProject list saved to './artifacts/lithium_project_list.csv'")

# Display the number of unique projects
print(f"\nTotal number of unique projects: {df['project'].nunique()}")

# Result accumulation storage section
from datetime import datetime

# Create artifacts directory
os.makedirs('./artifacts', exist_ok=True)

# Result file path
results_file = './artifacts/all_results.txt'
backup_file = './artifacts/all_results_backup_{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# Current analysis parameters
stage_name = "리튬 프로젝트 데이터 분석 및 전처리"
result_description = f"""리튬 프로젝트 데이터 분석 결과:
- 총 프로젝트 수: {len(df)}
- 고유 프로젝트 수: {df['project'].nunique()}
- 지역별 분포: {dict(region_counts)}
- 상태별 분포: {dict(status_counts)}
- 자원 유형별 분포: {dict(resource_counts)}
- 추출 방법별 분포: {dict(extraction_counts)}
- 상위 10개 국가별 프로젝트 수: {dict(country_counts)}
- 한계원가(Total Cash Cost) 통계:
  - 평균: {df['Total_Cash_Cost'].mean():.2f}
  - 중앙값: {df['Total_Cash_Cost'].median():.2f}
  - 최소값: {df['Total_Cash_Cost'].min():.2f}
  - 최대값: {df['Total_Cash_Cost'].max():.2f}
- 지역별 평균 한계원가: {dict(region_cost.round(2))}
"""

artifact_files = [
    ["./artifacts/projects_by_region.png", "지역별 리튬 프로젝트 분포 시각화"],
    ["./artifacts/avg_cost_by_region.png", "지역별 평균 한계원가 시각화"],
    ["./artifacts/projects_by_status.png", "프로젝트 상태별 분포 시각화"],
    ["./artifacts/lithium_project_list.csv", "리튬 프로젝트 리스트 (CSV)"]
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