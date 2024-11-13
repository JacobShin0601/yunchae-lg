import time
import boto3
import httpx
import numpy as np
from typing import Union, List

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain_anthropic import ChatAnthropic
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import BedrockEmbeddings

from .utils import remove_before_substring, cosine_similarity
import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel, Field
import json


class Main_Extract(BaseModel):
    main: str = Field(
        description = '''
        (반드시 있어야 함) 핵심내용 한줄 요약 서술 
        ''',
        defalut='핵심 내용이 없습니다.',
    )


class Content_Extract(BaseModel):
    content: str = Field(
        description = '''
        (반드시 있어야 함) 주요내용에 대해 구체적인 숫자를 근거로 자세한 내용 작성
        ''',
        defalut='요약 내용이 없습니다.',
    )


class Condition_Extract(BaseModel):
    condition: str = Field(
        description = '''
        (반드시 있어야 함) 텍스트를 읽고 시장 상황에 대한 판단
        ''',
        enum=["긍정", "부정", "중립"],
        defalut='중립',
    )


class Reason_Extract(BaseModel):
    reason: str = Field(
        description = '''
        (반드시 있어야 함) 해당 평가에 대한 이유
        ''',
        defalut='근거가 부족합니다.',
    )


class Time_Extract(BaseModel):
    time: int = Field(
        description = '''
        (반드시 있어야 함) 글을 읽는 예상 시간
        '''
    )


class AnthropicLangChain:
    def __init__(self, api_key:str, model_id:str, **kwargs):
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        self.model_id = model_id
        self.kargs = kwargs
        self.model_kwargs =  {
            "max_tokens": kwargs['config']['max_tokens'],
            "temperature": kwargs['config']['temperature'],
        }
        self.llm= ChatBedrock(
            client=bedrock_runtime,
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
        )
        
    def get_map_and_incremental_enhancing(self, dense_summary: str, sorted_news: dict, num_news=5) -> str:
        num_news = min(num_news, len(sorted_news))
        news_summaries = []

        for idx in range(num_news):
            news_summaries.append(sorted_news[idx]['summary'])
        total_summaries = '\n\n'.join(news_summaries)
        
        template = '''
            Below is a summary of an article (news_summary) and a detailed summary (dense_summary) that needs further enhancement. Use the instructions provided to refine and improve the dense summary.

            [Task Instructions]
            1. **Identify Related Information:** Analyze {news_summary} to identify any key facts, entities, or details that are relevant and missing from {dense_summary}.
            2. **Enhance Summary:** Modify and amend these relevant details to {dense_summary} where appropriate, while keeping the writing concise and logical.
            3. **Maintain Clarity and Accuracy:** Ensure that the final version is clear, accurate, and retains the structure of {dense_summary}.
            4. **Retain Non-related Content:** If {news_summary} does not contain any useful information, retain {dense_summary} as-is.
            5. **Tone down or remove strong and provocative language:** For example, “Ford EVs are doomed”.

            Here's the article summary (news_summary):
            {news_summary}

            And this is the detailed summary (dense_summary) that needs enhancing:
            {dense_summary}

            Produce a revised, comprehensive summary that combines the relevant details:
            '''
        prompt = PromptTemplate(template=template, input_variables=['dense_summary', 'news_summary'])
        chain = prompt | self.llm
        enhanced_dense_summary = chain.invoke({'dense_summary': dense_summary, 'news_summary': total_summaries}).content
        return enhanced_dense_summary
    
    def get_content(self, text, subject, try_cnt=20):
        client = instructor.from_anthropic(AnthropicBedrock())
        model_name = self.model_id
        max_token = self.model_kwargs['max_tokens']
        temperature = self.model_kwargs['temperature']
        entity = {}

        for idx in range(try_cnt):
            try:
                res_main = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_token,
                    temperature=temperature,
                    response_model=Main_Extract,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 세계 최고 수준의 전기차와 배터리 관련 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f'''
                            <요약 내용>
                            : {text}
            
                            <가이드라인>
                            1. 고유명사는 표기 언어 그대로 작성해주세요.
                            2. 구체적인 숫자 정보가 있다면 해당 정보를 기반으로 자세한 내용을 작성해주세요.
                            3. 언어의 표현은 deterministic하지 않아도 됩니다. (예: "약 100명" -> "약 100명 정도")
                            4. 특정 사람이 아닌 일반적인 사람들에게 해당하는 정보를 요약해주세요.
            
                            <지시사항>
                            (반드시 있어야 함) 요약 내용에 대해 가이드라인에 따라 {subject}에 관련된 핵심 내용을 한줄 헤드라인으로 만들어줘.
                            '''
                        },
                    ],
                ) 
                entity['주요내용'] = json.loads(res_main.model_dump_json())['main']
                break
            except Exception as e:
                continue
            
        for idx in range(try_cnt):
            try:            
                res_content = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_token,
                    temperature=temperature,
                    response_model=Content_Extract,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 세계 최고 수준의 전기차와 배터리 관련 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f'''
                            <주요 내용>
                            : {entity['주요내용']}
                            
                            <요약 내용>
                            : {text}
                
                            <가이드라인>
                            1. 고유명사는 표기 언어 그대로 작성해주세요.
                            2. 구체적인 숫자 정보가 있다면 해당 정보를 기반으로 자세한 내용을 작성해주세요.
                            3. 언어의 표현은 deterministic하지 않아도 됩니다. (예: "약 100명" -> "약 100명 정도")
                            4. 특정 사람이 아닌 일반적인 사람들에게 해당하는 정보를 요약해주세요.
                
                            <지시사항>
                            (반드시 있어야 함, 불렛포인트 형식으로) 주요 내용에 대해 요약 내용을 근거로 내용 작성
                            '''
                        },
                    ],
                )  
                entity['내용'] = json.loads(res_content.model_dump_json())['content'].replace("• ", "").split("\n\n")
                break
            except Exception as e:
                continue

        for idx in range(try_cnt):
            try:            
                res_condition = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_token,
                    temperature=temperature,
                    response_model=Condition_Extract,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 세계 최고 수준의 전기차와 배터리 관련 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f'''
                            <요약 내용>
                            : {text}
                            
                            <지시사항>
                            (반드시 있어야 함) 요약 내용에 기반하여 {subject}에 긍정/부정/중립 인지 판단해주세요.
                            '''
                        },
                    ],
                )  
                entity['긍부정 평가'] = json.loads(res_condition.model_dump_json())['condition']
                break
            except Exception as e:
                continue
                
        for idx in range(try_cnt):
            try:                            
                res_reason = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_token,
                    temperature=temperature,
                    response_model=Reason_Extract,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 세계 최고 수준의 전기차와 배터리 관련 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f'''
                            <주요 내용>
                            : {entity['주요내용']}
            
                            <내용>
                            : {entity['내용']}
            
                            <긍부정 평가>
                            : {entity['긍부정 평가']}
                            
                            <가이드라인>
                            1. 고유명사는 표기 언어 그대로 작성해주세요.
                            2. 구체적인 숫자 정보가 있다면 해당 정보를 기반으로 자세한 내용을 작성해주세요.
                            3. 언어의 표현은 deterministic하지 않아도 됩니다. (예: "약 100명" -> "약 100명 정도")
                            4. 특정 사람이 아닌 일반적인 사람들에게 해당하는 정보를 요약해주세요.
                        
                            <지시사항>
                            (반드시 있어야 함) 그리고 {subject}에 대해 긍부정 평가를 했는데 그 이유에 대해 
                            주요 내용과 내용을 기반하여 가이드라인에 따라 두줄 이내로 답변해주세요.
                            '''
                        },
                    ],
                )  
                entity['긍부정 평가'] += f". {json.loads(res_reason.model_dump_json())['reason']}"
                break
            except Exception as e:
                continue

        for idx in range(try_cnt):
            try:                  
                res_time = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_token,
                    temperature=temperature,
                    response_model=Time_Extract,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 세계 최고 수준의 전기차와 배터리 관련 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f'''
                            <주요 내용>
                            : {entity['주요내용']}
            
                            <내용>
                            : {entity['내용']}
            
                            <긍부정 평가>
                            : {entity['긍부정 평가']}
            
                            <지시사항>
                            (반드시 있어야 함) 주요 내용, 내용, 긍부정 평가를 읽는데 소요되는 시간을 답변해주세요.
                            시간은 '초' 단위로 말해주세요.
                            '''
                        },
                    ],
                )
                entity['시간'] = json.loads(res_time.model_dump_json())['time']
                break
            except Exception as e:
                continue
        return entity
        
    def get_intend(self, text:str, subject:str='electric vehicle') -> str:
        """get_intend _summary_

        Args:
            text (str): _description_
            args (_type_): _description_

        Returns:
            str: _description_
        """

        template = """

        <paper>
        {text}
        </paper>

        [Task instructions]
        here is a paper above.
        read the paper and determine whether the content is related to the {subject}. \
        If the content is related to the {subject}, please **just** say ‘yes’,
        IF the content is not related to the {subject}, please **just** say ‘no’. \
        You must follow this format. \

        result: ‘yes’ \
        result: ‘no’ \

        result:
        """

        prompt = PromptTemplate(template=template, input_variables=['subject', 'text'])
        chain = prompt | self.llm
        intend = chain.invoke({'subject':subject, 'text':text}).content
        return intend.lower().replace(' ','').replace('result:','')

    def get_article_summary(self, text:str, subject:str='전기차 시장') -> str:
        template = '''
        Here is a article.
        Read the article carefully and summarise the information related to {subject} using the [Task instructions]:

        [Task instructions]
        1. if you have specific numerical information, please use it in your summary..
        2. Use proper nouns in their original language.
        3. Summarise the information related to {subject}, focusing on the main points.
        4. Only the years 2024, 2025, 2026, 2027, and 2028 are covered in the summary.

        <paper>
        {text}
        </paper>
        '''
        prompt = PromptTemplate(template=template, input_variables=['subject', 'text'])
        chain = prompt | self.llm
        res = chain.invoke({'subject':subject, 'text':text})
        return res

    def chain_of_density_summary(self, article, subject:str='전기차 시장', try_cnt=20):
        prompt = '''
        Article: {ARTICLE}

        Your role is a reporter covering the automotive and battery sectors.
        You will generate increasingly concise, entity-dense summaries of the above article.

        Repeat the following 2 steps 5 times.
        
        Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
        Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

        A missing entity is:
        - Relevant : to the main story.
        - Specific : descriptive yet concise (5 words or fewer).
        - Novel : not in the previous summary.
        - Faithful : present in the Article.
        - Anywhere : located anywhere in the Aricle.

        Guidelines:
        - The summaries, summarise the information related to {SUBJECT}.
        - Only the years 2024, 2025, 2026, 2027, and 2028 are covered in the summary.
        - The first summary should be long (8-10 sentences, 180~200 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach 180~200 words.
        - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
        - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
        - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

        Remember, use the exact same number of words for each summary.
        Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
        Answer in JSON without necessarily adding any introductory text or meta comments.
        '''
        for idx in range(try_cnt):
            cod_prompt = ChatPromptTemplate.from_template(prompt)
            chain = cod_prompt | self.llm | JsonOutputParser()
            try:
                return chain.invoke({"ARTICLE": article, "SUBJECT": subject})
            except Exception as e:
                continue
        return "CoD Fail"
        

    def check_summary_if_related_for_news(self, denser_summary, news_article_list, key):
        data_list = []
        embeddings_client = BedrockEmbeddings()
        cod_embedding = embeddings_client.embed_query(denser_summary)
        
        for news in news_article_list:
            news['text'] = news['text'][:500000]
            if (news['text'] == '') or (len(news['text']) < 10):
                news['intend'] = 'no'
                news['summary'] = 'no'
                news['cosine_similarity'] = 0
            else:
                try:
                    news_intend = self.get_intend(subject = key, text = news['text'])
                except:
                    news_intend = 'no'
                if 'no' in news_intend :
                    news['intend'] = 'no'
                    news['summary'] = 'no'
                    news['cosine_similarity'] = 0
                else:
                    news['intend'] = 'yes'
                    news['summary'] = self.get_article_summary(subject = key, text = news['text']).content
                    news_embedding = embeddings_client.embed_query(news['summary'])
                    news['cosine_similarity'] = cosine_similarity(cod_embedding, news_embedding)
            data_list.append(news)
        return sorted(data_list, key=lambda x: x['cosine_similarity'], reverse=True)

    def filter_cod_summaries(self, dict_denser_summaries_and_missing_entities: dict, lst_all_news: list):
        filtered_summaries_dict = {}

        for i, (denser_summaries, missing_entities, _) in dict_denser_summaries_and_missing_entities.items():
            filtered_summaries = []
            if self.check_summary_if_related_for_news_titles(denser_summaries[-1], lst_all_news):
                filtered_summaries.append(denser_summaries[-1])
            else:
                print(f"Unrelated denser summary: {denser_summaries[-1]}")
            filtered_summaries_dict[i] = filtered_summaries
        return filtered_summaries_dict

    def get_summary_enhanced_with_news_titles(self, denser_summary:str, news_titles:list):
        combined_titles = ', '.join(news_titles)
        template = """
        Denser Summary: {denser_summary}
        News Titles: {combined_titles}

        [Task instructions]
        Enhance the provided denser summary by incorporating relevant information from the news titles.

        Enhanced Summary:
        """
        prompt = PromptTemplate(template=template, input_variables=['denser_summary', 'combined_titles'])
        llm = ChatAnthropic(**self.model_args)
        chain = LLMChain(llm=llm, prompt=prompt)
        enhanced_summary = chain.invoke({'denser_summary': denser_summary, 'combined_titles': combined_titles})['text']
        enhanced_summary = enhanced_summary.split('\n', 1)[-1]

        return enhanced_summary

    def get_summary_evaluation_score(self, article:str, denser_summary:str):
        template = """
        Article: {article}
        Summary: {denser_summary}
        Please rate the summary (1=worst to 5=best) with respect to below [Evaluation Criteria].

        [Evaluation Criteria]
        Below, we present the definitions provided for each quality metric. Please carefully consider each criterion and assign a score accordingly.
        •  Informative: An informative summary effectively captures the most important information in the article and presents it accurately and concisely. Rate the summary based on how well it covers essential information and avoids unnecessary details or omissions.
        •  Quality: A high quality summary is exceptionally clear, comprehensible, and free of ambiguity. Rate the summary based on its clarity, coherence, and absence of ambiguity.
        •  Coherence: A coherent summary demonstrates a logical flow of ideas, well-structured organization, and seamless transitions between points. Rate the summary based on its structure, organization, and logical flow of ideas.
        •  Attributable: Is all the information in the summary fully attributable to the Article? Rate the summary based on how well it attributes all the information to the original article without introducing new ideas or concepts.
        •  Overall Preference: A good summary should not only convey the main ideas but also critically analyze and synthesize the content in a concise, logical, and coherent fashion. Rate the summary based on your overall preference and satisfaction with the summary, considering both its strengths and weaknesses.

        Answer in JSON. The JSON should be a list dictionary with the key in above [Evaluation Criteria] and the value as the score (1-5).

        For example:
        [
            {
                "Informative": 5,
                "Quality": 5,
                "Coherence": 5,
                "Attributable": 5,
                "Overall Preference": 5
            }
        ]
        """
        prompt = PromptTemplate(template=template, input_variables=['article', 'denser_summary'])
        llm = ChatAnthropic(**self.model_args)
        chain = LLMChain(llm=llm, prompt=prompt)
        evaluation_score = chain.invoke({'article': article, 'denser_summary': denser_summary})['text']

        return evaluation_score

    def get_news_keywords(self, denser_summary: str, missing_entities: list) -> list:
        if not denser_summary.strip() or not any(entity.strip() for entity in missing_entities):
            print("Invalid input: Either the denser summary is empty or missing entities are not properly provided.")
            return []
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        template = """
        <paper>
        Denser Summary: {denser_summary}
        Missing Entities: {missing_entities}
        </paper>

        [Task instructions]
        Based on the provided denser summary and missing entities, please extract 3-5 detailed and descriptive keywords for searching relevant news articles on Google.
        Each keyword should contain multiple tokens and provide specific information related to the topic.
        Exclude any introductory messages or unnecessary information from your response.

        {format_instructions}
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['denser_summary', 'missing_entities'],
            partial_variables={"format_instructions": format_instructions}
        )
        chain = prompt | self.llm | output_parser
        keywords = chain.invoke({'denser_summary': denser_summary, 'missing_entities': missing_entities})
        return keywords