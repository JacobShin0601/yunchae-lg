import os
import re
import json
import multiprocessing

from tqdm import tqdm
from datetime import datetime, date
from pytz import timezone
from jinja2 import Environment, FileSystemLoader

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from .utils import dump_file, write_file, extract_text_from_tag
from .utils import remove_before_substring, extract_denser_summaries_and_missing_entities_and_original_transcript
from .utils import perform_google_news_search, fetch_article_info, get_random_user_agent
from .anthropic_api import AnthropicLangChain


class youtube_summary_wrapper:
    def __init__(self, youtube, args, config, key_config):
        self.youtube = youtube
        self.args = args
        self.config = config
        self.key_config = key_config
        self.sonnet_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.hauku_id = "anthropic.claude-3-haiku-20240307-v1:0"

    def _get_intend(self, text:str, key:str):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.hauku_id,
            config = self.config['LLM']
        )
        intend = anthropic.get_intend(
            text=text, 
            subject=key
        )
        return intend

    def _get_cod(self, article, subject, try_cnt=20):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.sonnet_id,
            config = self.config['LLM']
        )
        cod = anthropic.chain_of_density_summary(
            article=article, 
            subject=subject, 
            try_cnt=try_cnt
        )
        return cod

    def _get_event(self, item):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.hauku_id,
            config = self.config['LLM']
        )
        dict_denser_summaries_and_missing_entities = extract_denser_summaries_and_missing_entities_and_original_transcript(
            item=item
        )
        recommended_keywords = anthropic.get_news_keywords(
            denser_summary = dict_denser_summaries_and_missing_entities[0][-1],
            missing_entities = dict_denser_summaries_and_missing_entities[1][-1]
        )
        return recommended_keywords

    def _get_news(self, recommended_keywords, denser_summary, key, get_n=75):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.hauku_id,
            config = self.config['LLM']
        )
        google_news_result = perform_google_news_search(
            api_key = self.key_config['SERP_API']['key'], 
            base_query = key, 
            event_list = recommended_keywords,
            get_n=get_n,
        )
        headers = get_random_user_agent()
        new_list = fetch_article_info(google_news_result, headers)
        output = anthropic.check_summary_if_related_for_news(
            denser_summary = denser_summary, 
            news_article_list = new_list, 
            key = key
        )
        return output

    def _get_enhance(self, dense_summary, sorted_news, key):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.sonnet_id,
            config = self.config['LLM']
        )
        output = anthropic.get_map_and_incremental_enhancing(
            dense_summary=dense_summary,
            sorted_news=sorted_news
        )
        return output
    
    def _get_content(self, dense_summary, key, try_cnt=20):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            model_id = self.sonnet_id,
            config = self.config['LLM']
        )
        output = anthropic.get_content(
            text=dense_summary, 
            subject=key
        )
        return output

    def _get_infos(self, target_date):
        if target_date == 'now':
            _videos = str(date.today().strftime('%Y-%m-%d') + '.json')
        else:
            _videos = str(target_date.strftime('%Y-%m-%d') + '.json')

        ## search videos from youtube
        if os.path.exists(_videos):
            with open(_videos, 'r', encoding='utf-8') as json_file:
                video_dict = json.load(json_file)
        else:
            video_dict = self.youtube.get_video_infos(self.args, self.config)

            for category in video_dict.keys():
                for video_info in video_dict[category]:
                    video_info['details'], video_info['duration'] = self.youtube.get_video_duration(video_info['id']['videoId'])
                    video_info['caption'] = self.youtube.get_caption(video_info['id']['videoId'])
                    video_info['statistics'] = self.youtube.get_statistics(video_info['id']['videoId'])
                    video_info['comments'] = self.youtube.get_comments(video_info['id']['videoId'])

            dump_file(video_dict, target_date, aws_env=self.args.aws_env)
            
        return video_dict
    
    def process_video_summary(self, shared_list, idx, video, key):
        video_id = video['id']['videoId']
        link = f'링크 : https://www.youtube.com/watch?v={video_id}\n'
        text = " ".join([item['text'] for item in video['caption']])
        text = text.replace("[Music]","")
        text = text.replace("[음악]","")
        text = text[:500000]

        if len(text) < 10:
            item = shared_list[idx]
            item['cod'] = "subtitle is not available"
            item['event'] = []
            item['news'] = []
            item['enhanced_raw_summary'] = "subtitle is not available"
            item['enhanced_summary'] = "subtitle is not available"
            shared_list[idx] = item
            return video

        intend = self._get_intend(text=text, key=key)

        if 'no' in intend :
            item = shared_list[idx]
            item['cod'] = "Not Related to Electric Car"
            item['event'] = []
            item['news'] = []
            item['enhanced_raw_summary'] = "Not Related to Electric Car"
            item['enhanced_summary'] = "Not Related to Electric Car"
            shared_list[idx] = item
            return video

        item = shared_list[idx]
        item['cod'] = self._get_cod(article=text, subject=key)
        
        if item['cod'] == None:
            # with open('cod_None.json', 'a', encoding='utf-8') as json_file:
            #     json.dump(item, json_file, ensure_ascii=False, indent=4)
            shared_list[idx] = item
            return video
            
        item['event'] = self._get_event(item=item)        
        item['news'] = self._get_news(
            recommended_keywords=item['event'], 
            denser_summary=item['cod'][-1]['Denser_Summary'], 
            key=key
        )
        item['enhanced_raw_summary'] = self._get_enhance(
            dense_summary=item['cod'][-1]['Denser_Summary'], 
            sorted_news=item['news'], 
            key=key
        )
        item['enhanced_summary'] = self._get_content(
            dense_summary=item['enhanced_raw_summary'], 
            key=key
        )
        shared_list[idx] = item
        
        return video

    def _filter_json(self, json_file):
    # 필터링된 데이터를 리스트로 유지
        filtered_data = []
        for item in json_file:
            try:
                if 'Not Related to' in item['cod'] or 'subtitle is not available' in item['cod']:
                    continue 
                else:
                    filtered_data.append(item)
            except:
                continue
        # with open('filtered_data.json', 'w', encoding='utf-8') as json_file:
        #     json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)
        return filtered_data
    
    def make_summary_dict_(self, jsonfile):
        summary = []
        cnt_dict = {}
        name_dict = {
            'gm EV': 'GM', 
            'Volkswagen EV': 'VW',
            'Ford EV': 'Ford',
            'Tesla EV': 'Tesla',
            'LITHIUM SUPPLY CHAIN': 'LITHIUM-SUPPLY-CHAIN',
            'LITHIUM INVESTMENT': 'LITHIUM-INVESTMENT',
            'battery "CATL"': 'CATL',
            '배터리 "삼성SDI"': '삼성SDI',
            '배터리 "SK On"': 'SK온',
            'electric vehicle policy':'EV-Policy',
            'EV market news': 'EV-News'
        }
        
        for query in list(jsonfile.keys()):
            temp = jsonfile[query]
            # dump json file
            # with open(f'{query}.json', 'w', encoding='utf-8') as json_file:
            #     json.dump(temp, json_file, ensure_ascii=False, indent=4)
            # print('dumped', query)

            # print(f'{query} : {len(temp)}')
            # print('------------------------------------')

            temp = self._filter_json(temp)
            # with open(f'{query}_filtered.json', 'w', encoding='utf-8') as json_file:
            #     json.dump(temp, json_file, ensure_ascii=False, indent=4)

            try:
                cnt_dict[name_dict[query].replace("-", "_")] = len(temp)
            except:
                cnt_dict[name_dict[query].replace("-", "_")] = 0
            
            for item in temp:
                item['category'] = name_dict[query]
                summary.append(item) 
                
        cnt_dict['oem_cnt'] = cnt_dict['GM'] + cnt_dict['VW'] + cnt_dict['Ford'] + cnt_dict['Tesla']
        cnt_dict['material_cnt'] = cnt_dict['LITHIUM_SUPPLY_CHAIN'] + cnt_dict['LITHIUM_INVESTMENT'] 
        cnt_dict['competitor_cnt'] = cnt_dict['CATL'] + cnt_dict['삼성SDI'] + cnt_dict['SK온'] 
        cnt_dict['environment_cnt'] = cnt_dict['EV_Policy'] + cnt_dict['EV_News']
            
        return summary, cnt_dict

    def make_summary_report(self, summary_list, cnt_dict, target_date, args):
        """summary_report _summary_

        Args:
            jsonfile (_type_): _description_
            date_before (int): _description_
            option (str): _description_

        Raises:
            ValueError: _description_
        """
        if target_date == 'now':
            seoul_tz = timezone('Asia/Seoul')
            current_time = datetime.now(seoul_tz)
        else:
            current_time = target_date

        # 템플릿 환경 설정
        env = Environment(loader=FileSystemLoader('.'))

        # 템플릿 로드
        template = env.get_template('src/format/report_template.html')
        current_year = current_time.year
        current_week = current_time.isocalendar()[1]
        
        output = template.render(
            year=current_year,
            week=current_week,
            summary=summary_list,
            cnt_dict=cnt_dict
        )

        # 결과 저장
        html_name = f'{current_time.strftime("%Y-%m-%d")} Youtube Trend Summary.html'
        write_file(html_name, output, aws_env=args.aws_env)

    def summary(self, video_dict, target_date):

        ## process of video summary
        processes = []
        for key in tqdm(video_dict.keys()):
            manager = multiprocessing.Manager()
            shared_list = manager.list(video_dict[key])

            for idx, vdict in enumerate(shared_list):
                process = multiprocessing.Process(target=self.process_video_summary, args=(shared_list, idx, vdict, key))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            video_dict[key] = list(shared_list)

        ## make dump log
        dump_file(video_dict, target_date, aws_env=self.args.aws_env)        
        summary_list, cnt_dict = self.make_summary_dict_(video_dict)
        
        return summary_list, cnt_dict

    def run(self, target_date):
        video_dict = self._get_infos(target_date)
        summary_list, cnt_dict = self.summary(video_dict, target_date)
        # with open('summary_list.json', 'w', encoding='utf-8') as json_file:
        #     json.dump(summary_list, json_file, ensure_ascii=False, indent=4)
        # print(summary_list)
        # print(cnt_dict)
        self.make_summary_report(summary_list, cnt_dict, target_date, self.args)

        return video_dict