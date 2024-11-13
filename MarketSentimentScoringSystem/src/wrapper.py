import os
import json
import multiprocessing

from tqdm import tqdm
from datetime import datetime, date
from pytz import timezone
from jinja2 import Environment, FileSystemLoader

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from .utils import dump_file, write_file, s3_upload
from .anthropic_api import AnthropicLangChain


class youtube_summary_wrapper:
    def __init__(self, youtube, args, config, key_config):
        self.youtube = youtube
        self.args = args
        self.config = config
        self.key_config = key_config

    def _get_intend(self, text:str, key:str):   
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            config = self.config['LLM']
        )
        intend = anthropic.get_intend(text, key)
        return intend
    
    def _get_cod(self, docs, key, try_cnt=20):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            config = self.config['LLM']
        )
        cod = anthropic.chain_of_density_summary(docs, key, try_cnt=try_cnt)
        return cod

    def _get_summary(self, docs, key:str):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            config = self.config['LLM']
        )
        summary = anthropic.get_summary(docs, key)
        return summary
    
    def _get_content(self, dense_summary, key, try_cnt=20):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            config = self.config['LLM']
        )
        output = anthropic.get_content(text=dense_summary, subject=key, try_cnt=try_cnt)
        return output
    
    def _get_total_summary(self, docs, key:str):
        anthropic = AnthropicLangChain(
            api_key = self.key_config['Anthropic']['key'],
            config = self.config['LLM']
        )
        summary, contents_list = anthropic.get_total_summary(docs, key)
        return summary, contents_list

    def _get_infos(self, target_date, args):
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

            dump_file(video_dict, aws_env=args.aws_env)
            
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
            item['summary'] = "subtitle is not available"
            item['HL_summary'] = "subtitle is not available"
            shared_list[idx] = item
            return video

        intend = self._get_intend(text, key)

        if 'no' in intend :
            item = shared_list[idx]
            item['cod'] = "Not Related to battery materials"
            item['summary'] = "Not Related to battery materials"
            item['HL_summary'] = "Not Related to battery materials"
            shared_list[idx] = item
            return video

        # 본문을 Chunk 단위로 쪼갬
        text_splitter = CharacterTextSplitter(
            chunk_size=300000,     # 쪼개는 글자수
            chunk_overlap=30000,   # 오버랩 글자수
            length_function=len,
        )

        text = link + text
        text = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in text]
        summary = self._get_summary(docs, key)
        
        cod = self._get_cod(text, key)
        item = shared_list[idx]
        item['cod'] = cod
        # print('line:', cod)
        HL_summary = self._get_content(item['cod'][-1]['Denser_Summary'], key)
        item = shared_list[idx]


        item['summary'] = summary
        item['HL_summary'] = HL_summary
        shared_list[idx] = item  
        # print(item)
        # print(video)

        return video

    def make_summary_dict_(self, jsonfile):
        """make_report_file _summary_

        Args:
            jsonfile (_type_): _description_

        Raises:
            ValueError: _description_
        """

        summary_list = []
        for q in jsonfile:
            if len(jsonfile[q]) != 0:
                summary_dict = {}

                summary_dict['name'] = q.upper()
                summary_dict['contents'] = []

                for v in jsonfile[q]:
                    # print(v)
                    
                    title = v['snippet']['title']
                    publishedAt = v['snippet']['publishedAt']
                    video_id = v['id']['videoId']
                    channeltitle = v['snippet']['channelTitle']
                    summary = v['summary']
                    HL_summary = v['HL_summary']
                    duration = v['duration']

                    if summary == "subtitle is not available":
                        continue
                    if summary == "no":
                        continue
                    if summary == "Not Related to battery materials":
                        continue

                    summary = summary.replace(' <guideline>','').replace('</guideline>','')
                    
                    main_idx = summary.find('주요내용 : ')
                    contents_idx = summary.find('\n\n내용')
                    summary_contents_idx = summary.find('- ')
                    ai_eval_idx = summary.find('에 대한 영향')

                    if (main_idx == -1) and (contents_idx == -1) and (ai_eval_idx == -1):
                        continue 

                    # main_summary =  summary[main_idx:contents_idx].replace('주요내용 :','') # + f'(업로드: {publishedAt})'
                    # summary_contents = summary[summary_contents_idx:ai_eval_idx].replace('수요/공급', '')
                    # summary_contents = summary_contents.split('- ')[1:]
                    # ai_eval = summary[ai_eval_idx:].replace('에 대한 영향','').replace(':','')                    
                    main_summary = HL_summary['주요내용']
                    if len(main_summary) <= 10:
                        main_summary = summary[main_idx:contents_idx].replace('주요내용 :','')

                    summary_contents = HL_summary['내용']
                    if len(summary_contents) <= 10:
                        summary_contents = summary[summary_contents_idx:ai_eval_idx].replace('수요/공급', '')
                        summary_contents = summary_contents.split('- ')[1:]

                    ai_eval = HL_summary['수요/공급에 대한 영향']
                    if len(ai_eval) <= 10:
                        ai_eval = summary[ai_eval_idx:].replace('에 대한 영향','').replace(':','')

                    contents_dict = {}

                    contents_dict['title'] = title + f' (업로드일 : {publishedAt[:10]})'
                    contents_dict['channeltitle'] = channeltitle
                    contents_dict['publishedAt'] = publishedAt
                    contents_dict['link'] = f'https://www.youtube.com/watch?v={video_id}'
                    contents_dict['thumbnail'] = f'https://img.youtube.com/vi/{video_id}/0.jpg'
                    contents_dict['summary'] = summary
                    contents_dict['main_summary'] = main_summary
                    contents_dict['summary_contents'] = summary_contents
                    contents_dict['ai_eval'] = ai_eval
                    contents_dict['duration'] = duration.replace(':', '')
                    
                    summary_dict['contents'].append(contents_dict)
            
                summary_list.append(summary_dict)
        # print(summary_list)
        return summary_list

    def process_total_summary(self, shared_list, video, key):
        contents_dict = {}

        if len(video[key]) != 0:
            tot_summary = ''
            for v in video[key]:
                # summary = v['HL_summary']['내용']
                # try:
                #     summary = v['cod'][-1]
                # except KeyError:
                #     continue
                summary = v['summary']

                if summary == "subtitle is not available":
                    continue
                elif summary == "Not Related to battery materials":
                    continue
                elif summary == "":
                    continue
                elif len(summary) <= 20:
                    continue
                # else:
                #     print(summary)
                #     summary = summary['Denser_Summary']
                
                tot_summary = tot_summary + summary + '\n\n---\n\n'
                
            # 본문을 Chunk 단위로 쪼갬
            text_splitter = CharacterTextSplitter(        
                separator="주요내용",
                chunk_size=10000,     # 쪼개는 글자수
                chunk_overlap=0,    # 오버랩 글자수
                length_function=len,
                is_separator_regex=True,
            )

            text = text_splitter.split_text(tot_summary)
            docs = [Document(page_content=t) for t in text]

            summary, contents_list = self._get_total_summary(docs, key)
            contents_dict["name"] = key.upper()
            contents_dict["keywords"] = contents_list

            shared_list.append(contents_dict)

        return shared_list

    def make_summary_report(self, summary_list, total_summary, target_date, args):
        """summary_report _summary_

        Args:
            jsonfile (_type_): _description_
            date_before (int): _description_
            option (str): _description_

        Raises:
            ValueError: _description_
        """
        if target_date == 'now':
            _time = datetime.now(timezone('Asia/Seoul'))
        else:
            _time = datetime

        # 템플릿 환경 설정
        env = Environment(loader=FileSystemLoader('.'))

        # 템플릿 로드
        template = env.get_template('src/report_format/report_template.html')

        # 오늘 날짜와 팀 이름 설정
        logo = "https://www.lgensol.com/assets/img/common/logo.svg"
        date_today = datetime.now().strftime("%Y-%m-%d")
        date_year = _time.isocalendar().year
        date_week = _time.isocalendar().week
        team_name = "CDO.AI/Big Data Center.AI Solution Department.AI Solution Team 1"

        # 템플릿 렌더링
        output = template.render(
            logo=logo,
            date_year=date_year,
            date_week=date_week,
            date_today=date_today,
            team_name=team_name,
            categories=total_summary,
            summary_list=summary_list
        )

        # 결과 저장
        html_name = f'{_time.strftime("%Y-%m-%d")} Youtube Trend Summary.html'
        write_file(html_name, output, aws_env=args.aws_env)
    

    def summary(self, video_dict, args):
        ## mprocess of video summary 
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
        dump_file(video_dict, aws_env=args.aws_env)
        summary_list = self.make_summary_dict_(video_dict)
    
        processes = []
        manager = multiprocessing.Manager()
        shared_list = manager.list([])

        for key in tqdm(video_dict.keys()):
            process = multiprocessing.Process(target=self.process_total_summary, args=(shared_list, video_dict, key))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        total_summary = list(shared_list)

        name_order = [
            "LITHIUM (PRICE | SUPPLY | DEMAND) FORECAST",
            "LITHIUM SUPPLY CHAIN",
            "LITHIUM INVESTMENT"
            ]
        total_summary = sorted(total_summary, key=lambda x: name_order.index(x['name']))

        return summary_list, total_summary
    
    def run(self, target_date):
        video_dict = self._get_infos(target_date, self.args)
        summary_list, total_summary = self.summary(video_dict, self.args)
        self.make_summary_report(summary_list, total_summary, target_date, self.args)

        return video_dict
