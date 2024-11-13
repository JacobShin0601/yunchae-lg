<hr style="height:3px; background-color:#ff69b4; border:none;">

# Key Projects for LG Energy Solution

This document provides an overview and description of key projects conducted for LG Energy Solution. Due to the **Non-disclosure Agreement (NDA)**, the actual code used in these projects cannot be shared. Instead, **Sample Code** based on similar research papers or concepts is provided to illustrate the approach and methodology used in each project.

---

## 1. Battery Material Price Forecasting (Lithium and Other Materials)

### Project Description
- **Objective**: Forecast the prices of key battery materials, including lithium, to support effective purchasing and strategic decision-making.
- **Market Sentiment Scoring System**: Developed a system that gathers data from various sources, such as analyst reports, 10-K filings, and YouTube, to extract insights on lithium market trends. Leveraging LLMs (Claude 3 Haiku and Claude 3.5 Sonnet), this system extracts information on both supply and demand aspects, which are then used as features and leading indicators in a 12-month forecasting model. Led the AWS architecture design for this project.
- **Long-term Lithium Price Forecasting Model**: In collaboration with LG AI Research, we are currently developing a five-year price forecasting model for lithium. The model is structured around an economic framework based on supply and demand, where parameters are fitted using deep learning techniques, achieving an 80% accuracy in backtesting. I specifically contributed by identifying key macroeconomic factors (liquidity and interest rate variables from the U.S. Treasury and Federal Reserve) and ensuring their statistical significance.

### Technologies Used
- **LLMs**: Claude 3 Haiku, Claude 3.5 Sonnet
- **Statistical Analysis**: Correlation and Granger causality tests, normality and homoscedasticity tests, and non-parametric tests
- **Data Sources**: Analyst & Research Institute reports, 10-K filings, YouTube (with future plans to add Reddit)
- **Tools**: Python, LangChain, Pydantic, AWS (Bedrock architecture design and deployment)

### Outcomes
- Successfully developed the Market Sentiment Scoring system, which serves as a key feature and leading indicator in the 12-month lithium price forecasting model.
- Achieved 80% backtest accuracy in the long-term price forecasting model in collaboration with LG AI Research, supporting future price predictions.

---

## 2. GenAI-Related Projects

### Project Description
- **YouTube Summarization System**: Developed a system to collect and summarize YouTube videos related to electric vehicle manufacturers, competitors, raw materials, and macroeconomic conditions. Contributed to significantly upgrading an existing prototype created by other team members. Utilized statistical analysis for keyword selection and enhanced content reliability by gathering relevant news articles. Referenced the "Chain of Density" paper to improve the quality of summaries and migrated the entire pipeline to AWS.
- **Multimodal RAG (Retrieval-Augmented Generation)**: Built a multimodal RAG system using the Claude 3.5 Sonnet model to integrate reports on electric vehicles and lithium. This system allows the model to interpret not only text but also time-series charts and images, generating captions as needed. Internal customers can search and review documents via chat-based queries, initially designed for the strategy and MI departments and later expanded to other departments. I conducted this project independently, and due to internal recognition, six additional derivative products were developed. The AWS architecture was designed for this project.

### Technologies Used
- **Summarization Models**: Statistical analysis for keyword selection, LLM-based text summarization (Chain of Density model)
- **Multimodal RAG**: Document preprocessing techniques, Claude 3.5 Sonnet model for integrating electric vehicle and lithium-related reports, vector storage
- **Tools**: Python, Unstructured.io, AWS (for summarizer pipeline migration and architecture design, OpenSearch, Lambda)

### Outcomes
- Successfully upgraded the YouTube summarization system, distributing weekly newsletters to internal stakeholders.
- Developed a multimodal RAG system that supports complex document retrieval and querying, used by various departments, resulting in the creation of six derivative products.

---

These projects contributed to strategic decision-making for LG Energy Solution and provided additional data insights and operational efficiencies. The Sample Code on GitHub is based on similar concepts and methodologies used in the actual projects.

   



<hr style="height:3px; background-color:#ff69b4; border:none;">

# Key Projects for LG Energy Solution

본 문서에는 LG 에너지솔루션에서 수행한 주요 프로젝트의 개요와 설명이 담겨 있습니다. **Non-disclosure Agreement (NDA)**'에 따라 실제 프로젝트에서 사용한 코드는 공개할 수 없으며, 동일한 논문이나 개념을 기반으로 한 **Sample Code**를 예시로 제공합니다. 이를 통해 각 프로젝트에서 사용된 접근 방식과 방법론을 이해하실 수 있습니다.

---

## 1. 배터리 소재 가격 예측 (리튬 및 기타 소재)

### 프로젝트 설명
- **목표**: 리튬을 포함한 주요 배터리 소재의 가격을 예측하여 효과적인 구매와 전략적 의사결정을 지원.
- **Market Sentiment Scoring 시스템**: 애널리스트 리포트, 기업의 10-K 보고서, YouTube와 같은 다양한 소스에서 데이터를 수집하여 리튬 시황에 대한 정보를 추출하는 시스템을 개발하였습니다. 수요와 공급 요인을 모두 고려해 LLM(Claude 3 Haiku 및 Claude 3.5 Sonnet)을 활용하여 데이터를 추출하였으며, 12개월 예측 모델의 feature이자 선행 지표로 사용했습니다. 이 프로젝트의 AWS 아키텍처 설계를 주도했습니다.
- **리튬 중장기 가격 예측 모델**: LG AI Research와 협업하여 리튬 가격의 5년 장기 예측 모델을 개발 중입니다. 수요 공급 기반의 경제 모델을 구성하여 파라미터를 딥러닝으로 피팅한 후, 가격 예측 모델을 설계하여 현재 백테스트 정확도 80%를 확보했습니다. 특히 저는 거시경제 기반의 주요 변수(미국 재무부 및 연준의 유동성 및 금리 관련 변수)를 발굴하고 통계적 유의성 확보에 기여했습니다.

### 사용 기술
- **통계 분석**: 상관관계 및 그레인저 인과성 검정, 정규성 및 등분산성 검정 및 비모수 검정
- **데이터 소스**: Analyst & Research Institute 리포트, 10-K 보고서, YouTube (향후 Reddit 추가 예정)
- **도구**: Python, LangChain, Pydantic, AWS (Bedrock 아키텍처 설계 및 배포)
- **Reference1(Youtube Summarizer)**: 
- **Reference2(Youtube Summarizer on article)**: https://www.mk.co.kr/news/business/10989892

### 성과
- Market Sentiment Scoring 시스템을 성공적으로 개발하여, 12개월 리튬 가격 예측 모델의 주요 feature이자 선행 지표로 활용.
- LG AI Research와의 협업을 통해 중장기 예측 모델에서 백테스트 정확도 80%를 달성하여 향후자가격 예측을 지원.

---

## 2. 생성형 AI 관련 프로젝트

### 프로젝트 설명
- **YouTube 요약 시스템**: 전기차 제조사, 경쟁사, 원자재 및 거시경제 시황과 관련된 유튜브 영상을 수집하고 요약하는 시스템을 개발하였습니다. 기존에 다른 인원이 제작한 프로토타입을 대대적으로 업그레이드하는데 기여했으며 키워드 선택에 통계 분석을 활용하고 영상과 유관한 뉴스기사를 수집해 유튜브 요약문에 대한 신뢰성을 강화하였습니다. Chain of Density 논문을 참조하여 요약문의 충실성을 개선하고 전체 파이프라인을 AWS로 이전하였습니다.
- **멀티모달 RAG (검색 강화 생성)**: Claude 3.5 Sonnet 모델을 활용하여 전기차 및 리튬 관련 보고서를 통합하는 RAG 시스템을 구축하였습니다. 단순히 텍스트뿐만이 아닌 시계열 차트나 그림 등도 LLM 모델이 읽고 캡션을 제작하는 방식으로 만들었습니다. 내부고객이 문서를 채팅 기반으로 검색하고 내용을 확인할 수 있도록 하여, 처음에는 전략 및 MI 부서에서 활용하였고 이후 다른 부서에도 전달되었습니다. 이 프로젝트는 전적으로 혼자 진행했으며 이후 내부적인 인정을 받아 6개의 파생 프로덕트로 제작되었으며, AWS로 아키텍처를 구성하였습니다.

### 사용 기술
- **요약 모델**: 키워드 선택을 위한 통계 분석, LLM 기반 텍스트 요약 모델 (Chain of Density 모델).
- **멀티모달 RAG**: 문서 전처리 기술 & 전기차 및 리튬 관련 보고서 통합을 위한 Claude 3.5 Sonnet 모델 & 벡터 스토
- **도구**: Python, Unstructured.io, AWS(서머라이저 파이프라인 이전 및 아키텍처 설계, OpenSearch, Lambda...).

### 성과
- YouTube 요약 시스템을 성공적으로 업그레이드하여 내부 이해관계자들에게 주간 뉴스레터 형태로 배포.
- 멀티모달 RAG 시스템을 개발하여 복잡한 문서의 검색과 조회를 지원하고, 다양한 부서에서 활용되었으며 6개의 파생 제품이 추가로 제작됨.

---

이 프로젝트들은 LG 에너지솔루션의 전략적 의사결정에 기여했으며, 추가적인 데이터 인사이트와 운영 효율성을 제공했습니다. GitHub의 Sample Code는 실제 프로젝트에서 사용된 개념과 방법론을 바탕으로 작성되었습니다.
