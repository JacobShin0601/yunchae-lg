<hr style="height:3px; background-color:#ff69b4; border:none;">

# Key Projects for LG Energy Solution

This document provides an overview and description of key projects conducted for LG Energy Solution. Due to the **Non-disclosure Agreement (NDA)**, the actual code used in these projects cannot be shared. Instead, **Sample Code** based on similar research papers or concepts is provided to illustrate the approach and methodology used in each project.

---

## 1. Battery Material Price Forecasting (Lithium and Other Materials)

### Project Description
- **Short-term Price Forecasting (1-year, Spot Purchasing Optimization)**: Built an enhanced short-term model to optimize purchasing timing for the 20–30% spot volume beyond long-term contracts, aiming at cost reduction.
- **Role**: Served as the sole LLM application developer in a 10-person materials data science team. Designed and implemented a pipeline that collects, cleans, and converts unstructured lithium/materials news, reports, and analyst commentary into features for time-series price models.
- **Engineering**: Delivered Python crawlers, LLM-based extraction (summarization/tagging), and ETL workflows in a production-deployable form.
- **Mid-to-Long-term Price Outlook & Mine Investment Feasibility (5-year horizon)**: Core member of a joint task force (LG AI Research, internal data science, and metals business development). Led supply-side modeling and simulator design, structured 10-K (EDGAR) data into inventory/Capex/production proxies, built supply scenarios (base/tight/overshoot), and designed a deep-learning price module reflecting bubble/overheating regimes.

### Technologies Used
- **LLMs**: Claude 3 Haiku, Claude 3.5 Sonnet
- **Data Sources**: Global news, analyst reports, 10-K filings (EDGAR), FRED dataset
- **Analysis & Modeling**: Feature engineering from unstructured text, supply-side scenario simulation, deep-learning price outlook
- **Tools**: Python, LangChain, Pydantic, AWS (LLMOps by Bedrock architecture design and deployment)

### Reference
- **Reference1(Sentiment Scoring System)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MarketSentimentScoringSystem
- **Reference2(Graph Extraction Paper)**: https://arxiv.org/html/2404.16130v1
- **Reference3(Graph Extraction Github by Microsoft)**: https://github.com/microsoft/graphrag 

### Outcomes
- The short-term model provided quantitative inputs for spot purchasing decisions, contributing to ~KRW 12B annual cost savings and improved forecast error over baseline time-series models by combining structured price/inventory with market sentiment.
- The 5-year outlook supported IRR/NPV evaluation for mine equity investments and offtake negotiations; internally assessed to contribute up to ~KRW 300B over five years, with a 10% contribution officially attributed to this workstream.

---

## 2. GenAI-Related Projects

### Project Description
- **YouTube Summarization System**: Developed a system to collect and summarize YouTube videos related to electric vehicle manufacturers, competitors, raw materials, and macroeconomic conditions. Contributed to significantly upgrading an existing prototype created by other team members. Utilized statistical analysis for keyword selection and enhanced content reliability by gathering relevant news articles. Referenced the "Chain of Density" paper to improve the quality of summaries and migrated the entire pipeline to AWS. News letter was deployed to the whole company weekly.
- **Multimodal RAG (Retrieval-Augmented Generation)**: Built a multimodal RAG system using the Claude 3.5 Sonnet model to integrate reports on electric vehicles and lithium. This system allows the model to interpret not only text but also time-series charts and images, generating captions as needed. Internal customers can search and review documents via chat-based queries, initially designed for the strategy and MI departments and later expanded to other departments. I conducted this project independently, and due to internal recognition, six additional derivative products were developed. The AWS architecture was designed for this project.

### Technologies Used
- **Summarization Models**: Statistical analysis for keyword selection, LLM-based text summarization (Chain of Density model)
- **Multimodal RAG**: Document preprocessing techniques, Claude 3.5 Sonnet model for integrating electric vehicle and lithium-related reports, vector storage
- **Tools**: Python, Unstructured.io, AWS (for summarizer pipeline migration and architecture design, OpenSearch, Lambda), STREAMLIT

### Reference
- **Reference1(Youtube Summarizer)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/YoutubeSummarizer
- **Reference2(Youtube Summarizer on article)**: https://www.mk.co.kr/news/business/10989892
- **Reference3(Chain of Density paper)**: https://arxiv.org/abs/2309.04269 
- **Reference4(Multimodal RAG)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MultimodalRAG 

### Outcomes
- Successfully upgraded the YouTube summarization system, distributing weekly newsletters to internal stakeholders.
- Developed a multimodal RAG system that supports complex document retrieval and querying, used by various departments, resulting in the creation of six derivative products.

---

## 3. Multi-Agent Research Workflow (Market Intelligence)

### Project Description
- **Objective**: Build a multi-agent research pipeline that gathers, analyzes, and summarizes global lithium project signals (production cuts, closures, new developments) into structured reports.
- **Workflow**: Coordinator → Planner → Supervisor → Researcher (search/crawl) → Coder (data analysis) → Reporter (final report). The pipeline produces Markdown/HTML reports and visual artifacts for internal review.
- **Outputs**: Example deliverables include `final_report.md`, progressive reports, and charts generated in `artifacts/`.

### Technologies Used
- **Agents & Orchestration**: Async workflow, role-based prompts (researcher/coder/reporter)
- **Data & Analysis**: Python, pandas-based analysis, visualization artifacts
- **Tools**: Search/crawl utilities (Tavily), report generation to Markdown/HTML

### Reference
- **Reference1(Multi-Agent Research)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MultiAgentResearch

### Outcomes
- Produced structured lithium project intelligence reports and reusable research templates.

---

## 4. PCF Optimization System (Battery Manufacturing)

### Project Description
- **Objective**: Optimize battery manufacturing inputs to reduce Product Carbon Footprint (PCF) while balancing cost.
- **Optimization Scope**: Cathode/anode composition, recycled and low-carbon materials, RE100 adoption, and Pareto trade-off exploration.
- **Interface**: Streamlit-based UI for simulation, constraint setup, solver selection, and results comparison.

### Technologies Used
- **Optimization**: GLPK/IPOPT solvers, Pareto methods (Weighted Sum, Epsilon-Constraint, NSGA-II)
- **App & Analysis**: Python, Streamlit, modular optimization engine

### Reference
- **Reference1(PCF Optimization)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/PCF_sim_opt

### Outcomes
- Enabled scenario-based evaluation of carbon vs. cost trade-offs with configurable constraints.

---

These projects contributed to strategic decision-making for LG Energy Solution and provided additional data insights and operational efficiencies. The Sample Code on GitHub is based on similar concepts and methodologies used in the actual projects.

   



<hr style="height:3px; background-color:#ff69b4; border:none;">

# Key Projects for LG Energy Solution

본 문서에는 LG 에너지솔루션에서 수행한 주요 프로젝트의 개요와 설명이 담겨 있습니다. **Non-disclosure Agreement (NDA)**'에 따라 실제 프로젝트에서 사용한 코드는 공개할 수 없으며, 동일한 논문이나 개념을 기반으로 한 **Sample Code**를 예시로 제공합니다. 이를 통해 각 프로젝트에서 사용된 접근 방식과 방법론을 이해하실 수 있습니다.

---

## 1. 배터리 소재 가격 예측 (리튬 및 기타 소재)

### 프로젝트 설명
- **단기 가격 예측 고도화 (1년, Spot 구매 최적화)**: 선도계약 외 20~30% Spot 물량의 구매 타이밍 최적화를 통해 원가 절감을 목표로 한 단기 예측 모델 고도화.
- **역할**: 원자재 데이터 사이언티스트 10인 팀 내 LLM 응용 개발 담당 1인으로 참여. 글로벌 리튬·원자재 뉴스/리포트/애널리스트 코멘트 등 비정형 텍스트를 수집·정제하고 시계열 가격모델의 Feature로 자동 반영하는 파이프라인 설계 및 구현.
- **엔지니어링**: Python 기반 크롤러, LLM 기반 정보 추출(요약/태깅), ETL 워크플로우를 운영환경 배포 가능 형태로 구축.
- **리튬 중장기 가격 전망 및 광산투자 타당성 분석 (5년 Horizon)**: LG AI연구원·내부 데이터 사이언티스트·메탈사업개발팀 합동 태스크포스 코어 멤버로 참여. 공급 사이드 모델링 및 시뮬레이터 총괄, 10-K(EDGAR) 공시를 크롤링해 재고/Capex/생산계획 Proxy 구조화, 공급 시나리오(base/tight/overshoot) 설계, 과열·버블 구간을 반영하는 딥러닝 기반 가격 전망 모듈 설계.

### 사용 기술
- **LLM들**: Claude 3 Haiku, Claude 3.5 Sonnet
- **데이터 소스**: 글로벌 뉴스, 애널리스트 리포트, 10-K(EDGAR), FRED dataset
- **분석/모델링**: 비정형 Feature 엔지니어링, 공급 시나리오 시뮬레이션, 딥러닝 가격 전망
- **도구**: Python, LangChain, Pydantic, AWS (Bedrock 아키텍처 설계 및 배포를 통한 LLMOps)

### Reference
- **Reference1(Sentiment Scoring System)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MarketSentimentScoringSystem
- **Reference2(Graph Extraction Paper)**: https://arxiv.org/html/2404.16130v1
- **Reference3(Graph Extraction Github by Microsoft)**: https://github.com/microsoft/graphrag 

### 성과
- Spot 구매 의사결정의 정량 인풋으로 활용되며, 연간 약 120억 원 수준의 원가 절감에 기여.
- 정형(Price/Inventory)과 비정형(Market Sentiment) 결합으로 단순 시계열 대비 예측 오차율을 개선하고 의사결정 신뢰도 제고.
- 5년 전망 결과를 기반으로 광산 지분투자 및 오프테이크 협상 시 IRR/NPV 검토 자료로 활용되었고, 향후 5년 약 3,000억 원 수준의 이익 기여 가능 프로젝트로 평가(기여도 10% 공식 인정).

---

## 2. 생성형 AI 관련 프로젝트

### 프로젝트 설명
- **YouTube 요약 시스템**: 전기차 제조사, 경쟁사, 원자재 및 거시경제 시황과 관련된 유튜브 영상을 수집하고 요약하는 시스템을 개발하였습니다. 기존에 다른 인원이 제작한 프로토타입을 대대적으로 업그레이드하는데 기여했으며 키워드 선택에 통계 분석을 활용하고 영상과 유관한 뉴스기사를 수집해 유튜브 요약문에 대한 신뢰성을 강화하였습니다. Chain of Density 논문을 참조하여 요약문의 충실성을 개선하고 전체 파이프라인을 AWS로 이전하였습니다. 개발 후 전사에 매주 메일로 배포 되고 있습니다.
- **멀티모달 RAG (검색 강화 생성)**: Claude 3.5 Sonnet 모델을 활용하여 전기차 및 리튬 관련 보고서를 통합하는 RAG 시스템을 구축하였습니다. 단순히 텍스트뿐만이 아닌 시계열 차트나 그림 등도 LLM 모델이 읽고 캡션을 제작하는 방식으로 만들었습니다. 내부고객이 문서를 채팅 기반으로 검색하고 내용을 확인할 수 있도록 하여, 처음에는 전략 및 MI 부서에서 활용하였고 이후 다른 부서에도 전달되었습니다. 이 프로젝트는 전적으로 혼자 진행했으며 이후 내부적인 인정을 받아 6개의 파생 프로덕트로 제작되었으며, AWS로 아키텍처를 구성하였습니다.

### 사용 기술
- **요약 모델**: 키워드 선택을 위한 통계 분석, LLM 기반 텍스트 요약 모델 (Chain of Density 모델)
- **멀티모달 RAG**: 문서 전처리 기술 & 전기차 및 리튬 관련 보고서 통합을 위한 Claude 3.5 Sonnet 모델 & 벡터 스토어
- **도구**: Python, Unstructured.io, AWS(서머라이저 파이프라인 이전 및 아키텍처 설계, OpenSearch, Lambda...), Pydantic, LangChain

### Reference
- **Reference1(Youtube Summarizer)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/YoutubeSummarizer
- **Reference2(Youtube Summarizer on article)**: https://www.mk.co.kr/news/business/10989892
- **Reference3(Multimodal RAG)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MultimodalRAG 

### 성과
- YouTube 요약 시스템을 성공적으로 업그레이드하여 내부 이해관계자들에게 주간 뉴스레터 형태로 배포.
- 멀티모달 RAG 시스템을 개발하여 복잡한 문서의 검색과 조회를 지원하고, 다양한 부서에서 활용되었으며 6개의 파생 제품이 추가로 제작됨.

---

## 3. 멀티 에이전트 리서치 워크플로우 (마켓 인텔리전스)

### 프로젝트 설명
- **목표**: 글로벌 리튬 프로젝트의 감산, 폐쇄, 신규 개발 정보를 수집·분석·요약하는 멀티 에이전트 리서치 파이프라인 구축.
- **워크플로우**: Coordinator → Planner → Supervisor → Researcher(검색/크롤링) → Coder(데이터 분석) → Reporter(최종 보고서). Markdown/HTML 보고서와 시각화 산출물을 생성.
- **산출물**: `final_report.md`, progressive report, `artifacts/` 내 차트 및 리포트.

### 사용 기술
- **에이전트/오케스트레이션**: Async 워크플로우, 역할 기반 프롬프트
- **데이터 분석**: Python, 시각화 산출물
- **도구**: Tavily 기반 검색/크롤링, Markdown/HTML 리포트 생성

### Reference
- **Reference1(Multi-Agent Research)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/MultiAgentResearch

### 성과
- 리튬 프로젝트 인텔리전스 리포트와 재사용 가능한 리서치 템플릿을 제작.

---

## 4. PCF 최적화 시스템 (배터리 제조)

### 프로젝트 설명
- **목표**: 제품 탄소발자국(PCF)을 최소화하면서 비용과의 트레이드오프를 고려한 자재 구성 최적화.
- **최적화 범위**: 양극/음극 조성, 재활용 및 저탄소 소재, RE100 적용, 파레토 최적화 탐색.
- **인터페이스**: Streamlit 기반 UI로 시뮬레이션, 제약조건 설정, 솔버 선택, 결과 비교 제공.

### 사용 기술
- **최적화**: GLPK/IPOPT 솔버, 파레토 방법(Weighted Sum, Epsilon-Constraint, NSGA-II)
- **앱/분석**: Python, Streamlit, 모듈형 최적화 엔진

### Reference
- **Reference1(PCF Optimization)**: https://github.com/JacobShin0601/yunchae-lg/tree/main/PCF_sim_opt

### 성과
- 탄소-비용 트레이드오프를 시나리오로 비교·평가할 수 있는 최적화 파이프라인 구축.

---

이 프로젝트들은 LG 에너지솔루션의 전략적 의사결정에 기여했으며, 추가적인 데이터 인사이트와 운영 효율성을 제공했습니다. GitHub의 Sample Code는 실제 프로젝트에서 사용된 개념과 방법론을 바탕으로 작성되었습니다.
