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


class AnthropicLangChain:
    def __init__(self, api_key:str, **kwargs):
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        # model_id = "anthropic.claude-3-sonnet-20240229-v1:0" # Claude 3 Sonnet
        model_id = "anthropic.claude-3-haiku-20240307-v1:0" # Claude 3 Haiku
        model_kwargs =  {
            "max_tokens": kwargs['config']['max_tokens'],
            "temperature": kwargs['config']['temperature'],
        }
        self.llm= ChatBedrock(
            client=bedrock_runtime,
            model_id=model_id,
            model_kwargs=model_kwargs,
        )
        self.kargs = kwargs
        
    
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


    def get_content(self, text, subject, try_cnt=20):
        template = '''
        당신은 배터리 및 관련 광물 산업을 조사하는 연구원입니다.
        아래에 paper가 있습니다.
        paper의 내용을 자세히 읽고 {subject}와 관련된 내용에 대해서 [Task instructions]에 있는 <guideline>을 지켜서
        최대한 자세하게 korean으로 요약해줘.({subject}와 관련된 내용이 아니라면 "no"라고 대답해주세요.):
    
        <Task instructions>
        1. 고유명사물 표기 언어 그대로 작성해주세요.
        2. 구체적인 숫자 정보가 있다면 해당 정보를 기반으로 자세한 내용을 작성해주세요.
        3. {subject}와 관련된 내용을 메인으로 언급해주세요.
        4. 언어의 표현은 deterministic하지 않아도 됩니다. (예: "약 100명" -> "약 100명 정도")
        5. 특정 사람이 아닌 일반적인 사람들에게 해당하는 정보를 요약해주세요.
        </Task instructions>
    
        위의 지시사항들을 차례대로 생각해봅시다.
    
        <Guideline>
        주요내용 : (반드시 있어야 함) 핵심내용 한줄 요약 서술 \n
        내용 : \n
            (반드시 있어야 함, 불렛포인트 형식으로)주요내용에 대해 구체적인 숫자를 근거로 자세한 내용 작성 \n
        수요/공급에 대한 영향 : (반드시 있어야 함) 수요/공급 증가/감소, 평가에 대한 이유 \n
        </Guideline>

        다시 한번 말하지만, <Task instructions>을 지키고 <Guideline> 포맷에요따라 답변해주세요.

        답변은 JSON 형태로 만들어주세요. JSON에는 key값으로 "주요내용", "내용", "수요/공급에 대한 영향" 이어야만 합니다.
        
        <paper>
        {text}
        </paper>
        '''
        prompt = PromptTemplate(template=template, input_variables=['subject', 'text'])
        chain = prompt | self.llm | JsonOutputParser()
    
        for idx in range(try_cnt):
            try:
                return chain.invoke({'subject':subject, 'text':text})
            except:
                continue
        return {'주요내용':[], '내용':[], '수요/공급에 대한 영향':[]}    


    def chain_of_density_summary(self, article, subject:str='Lithium Market', try_cnt=20):
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
        '''

        cod_prompt = ChatPromptTemplate.from_template(prompt)
        chain = cod_prompt | self.llm | JsonOutputParser()

        for idx in range(try_cnt):
            try:
                return chain.invoke({"ARTICLE": article, "SUBJECT": subject})
            except:
                continue
        return None   


    def get_summary(self, text:str, subject:str='리튬 시장') -> str:
        """get_summary _summary_

        Args:
            text (str): _description_
            args (_type_): _description_

        Returns:
            str: _description_
        """

        # 각 Chunk 단위의 템플릿
        template = '''
        
        여기에 paper가 있습니다. paper의 내용을 자세히 읽고 다음의 내용을 요약해줘:

        <paper>
        {text}
        </paper>
        '''

        # 전체 문서(혹은 전체 Chunk)에 대한 지시(instruct) 정의
        combine_template = '''

        아래에 paper가 있습니다.
        paper의 내용을 자세히 읽고 {subject}와 관련된 내용에 대해서 [Task instructions] 아래에 있는 <guideline>을 지켜서 korean으로 요약해주세요.:

        [Task instructions]
        1. 고유명사가 있는 경우 영어로 표기해주세요.
        2. 구체적인 숫자를 기반으로 내용을 작성해 주세요.
        3. 리튬 시장에 관련된 수요/공급의 증가/감소에 대한 의견을 포함해주세요. 구체적인 수치가 있다면 더욱 좋아요.
        4. 수요/공급의 증가/감소에 대한 예시는 아래와 같아요.
            (예시) 수요/공급에 대한 영향 : 수요 증가 - 이유 // 공급 감소 - 이유
        5. 리튬을 제외한 금, 태양광 등에 관련된 내용은 제외해주세요.
        6. 요약의 결과는 아래의 <guideline>의 내용을 참고하여 요약을 작성해주세요.
        
        <guideline>
        주요내용 : 한 줄로 요약된 내용, \n
        내용 : \n
            (불렛포인트 형식으로)주요내용에 대해 구체적인 숫자를 근거로 내용 작성 \n
        수요/공급에 대한 영향 : (한줄에 작성하고 "리튬 시장"을 중심으로, 그리고 구체적인 수치가 있다면 수치도 함께 표시해주세요) '수요or공급 증가or감소' 혹은 '관련 없음' - 해당 평가에 대한 이유 (대답이 여러개라면 //을 넣어서 구분해주세요.) \n
        </guideline>

        <paper>
        {text}
        </paper>

        
        
        '''
        
        # 템플릿 생성
        prompt = PromptTemplate(template=template, input_variables=['subject', 'text'])
        combine_prompt = PromptTemplate(template=combine_template, input_variables=['subject', 'text'])

        llm = self.llm

        # 요약을 도와주는 load_summarize_chain
        chain = load_summarize_chain(llm, 
                                    map_prompt=prompt, 
                                    combine_prompt=combine_prompt, 
                                    chain_type='map_reduce', 
                                    verbose=False)
        
        res = chain.invoke({'subject':subject, 'input_documents':text})['output_text']
        # print(res)
        
        return res
    
        
    def get_total_summary(self, text:str, subject:str='리튬 시장') -> str:
        """get_summary _summary_
장
        Args:
            text (str): _description_
            args (_type_): _description_

        Returns:
            str: _description_
        """

        # 전체 문서(혹은 전체 Chunk)에 대한 지시(instruct) 정의
        template = '''
        
        {text}
        
        위에 요약 정보가 있습니다.
        당신은 전기차 시장 및 배터리 관련 산업을 조사하는 연구원입니다.
        특히, 배터리에 사용되는 리튬과 같은 광물 정보에 대해서 조사하고 있습니다.
        요약 정보를 내용을 자세히 읽고 {subject}과 관련된 내용들을 구체적인 숫자를 기반으로 정확하게 정리해주세요. :

        [Task instructions]
        0. 리튬시장과 관계없는 경우에는 요약을 하지 않아도 됩니다.
        1. **(반드시 지켜야함)** 요약의 결과는 아래의 <포맷>과 <예시>의 내용을 참고하여 요약을 작성해주세요.
        2. 요약 정보에서 {subject}에 대한 내용만 남겨주세요.
        3. 주요 내용 및 내용은 {subject}에 대한 내용만 남겨주세요.
        4. 주요내용은 여러개여도 상관없어요. 주요내용이 여러개인 경우 넘버링은 하지 마세요. 
        5. 고유명사의 경우 그 단어가 영어인 경우 영어로, 한국어인 경우 한국어로 표기해주세요.
        6. 구체적인 숫자를 기반으로 내용을 작성해 주세요.
        7. 날짜 정보가 기재되어 있다면 2024년 이후의 내용만 포함해주세요.
        8. 가급적 단어의 표현은 CEO 및 투자자가 읽을 수 있도록 부드러운 표현으로 만들어주면 좋겠어요.
        9. 다른 코멘트 없이 <포맷>의 형태로만 말해주면 됩니다.
        10. 위의 요구 사항들을 정확히 수행했는지 다시 한 번 확인해주세요.
        11. 리튬의 수요/공급에 대한 영향은 반드시 아래의 포맷을 지켜주시고 예시를 잘 참조해주세요.
        12. 리튬을 제외한 금, 태양광 등에 관련된 내용은 제외해주세요.

        <포맷>

        주요내용 : 한 줄로 요약된 내용 \n
        내용 :
            (작성 조건 : 1. 구체적인 숫자를 언급, 2. {subject}와 관련된 내용을 주로 언급, 3. 정책과 관련된 내용은 전부 언급) 
            작성 조건을 지키며 각 내용을 hypen형식으로 작성 \n
        수요/공급에 대한 영향 : (한줄에 작성하고 "리튬 시장"을 중심으로, 그리고 구체적인 수치가 있다면 수치도 함께 표시해줘) '수요or공급 증가or감소' 혹은 '관련 없음'을 명시하고 해당 평가에 대한 이유 \n
        --- 줄바꿈

        
        <예시>

        주요내용 : 주요 내용 1

        내용: - 내용1 - 내용2 - 내용3 - 내용4

        수요/공급에 대한 영향 : 전기차 보급 확대로 인한 리튬 수요 급증에 대응하기 위해서는 새로운 리튬 광산 개발이 필요할 것으로 보이며, 이에 따라 리튬 가격도 상승할 것으로 예상됨
        
        ---
        
        주요내용 : 주요 내용 2
        
        내용: - 내용1 - 내용2 - 내용3
        
        수요/공급에 대한 영향 : 공급 감소 - 이유Asai Cas의 투자로 인해 Port Colborne 지역이 전기차 공급망의 주요 투자 대상지로 부상할 것으로 예상되며, 이는 향후 리튬 등 배터리 관련 광물 수요 및 공급에 긍정적인 영향을 미칠 것으로 보임
        
        ---

        잠시 시간을 갖고 차근차근히 생각한 다음, 주어진 [Task instructions]를 <포맷>과 <예시>를 참고하여 차례대로 정확히 요약을 수행해주세요.
        

        '''
        
        prompt = PromptTemplate(template=template, input_variables=['subject', 'text'])

        llm = self.llm
        
        chain = load_summarize_chain(llm, 
                                    prompt=prompt, 
                                    chain_type='stuff', 
                                    verbose=False)
        res = chain.invoke({'subject':subject, 'input_documents':text})['output_text']
        
        contents_list = []

        for result in res.split('---'):
            err_flag = 0
            result_dict = {}

            if len(result) == 0:
                continue

            result = remove_before_substring(result, '주요내용 :')

            for line in result.split('\n\n'):
                if len(line) == 0:
                    continue

                try:
                    key, value = line.split(':',1)
                    key = key.removeprefix(' ').removesuffix(' ')
                    if key == '주요내용':
                        key = 'main_content'
                    elif key == '내용':
                        key = 'details'
                        value = value.split('- ')[1:]
                    elif key == '수요/공급에 대한 영향':
                        key = 'sentiment'
                    elif key == '영상 링크':
                        key = 'link'
                    result_dict[key] = value
                except:
                    print('<debug>')
                    print('---error line---')
                    print(line)
                    print('---total sentence---')
                    print(result)
                    print('</debug>')
                    err_flag = 1
                    continue

            if err_flag == 0:
                result_dict["keyword"] = subject.upper()
                contents_list.append(result_dict)

        return res, contents_list

    def chain_of_density_summary(self, article, subject:str='Lithium market', try_cnt=20):
        prompt = '''
        Article: {ARTICLE}

        Your role is a reporter covering battery materials sectors.
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
        '''

        cod_prompt = ChatPromptTemplate.from_template(prompt)
        chain = cod_prompt | self.llm | JsonOutputParser()

        for idx in range(try_cnt):
            try:
                return chain.invoke({"ARTICLE": article, "SUBJECT": subject})
            except:
                continue
        return None