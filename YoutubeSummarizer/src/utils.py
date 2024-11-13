import os
import glob
import platform
import json
import re
from datetime import datetime, date, timedelta
from jinja2 import Environment, FileSystemLoader
from pytz import timezone
import requests
from bs4 import BeautifulSoup
import time
import random
import serpapi
import numpy as np


def get_config():
    # Read config file
    config_path = "config.json"
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)

    return config


def get_key_config():
    import boto3
    
    # Read config file
    bucket = 'content-summarizer'
    key = 'key.json'
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket = bucket, Key = key)
    config = json.loads(data['Body'].read())

    return config


def get_rfc_date(date= datetime.now(timezone('Asia/Seoul')).today(), before_days:int=15):
    # within one month
    date_before = date - timedelta(before_days)
    rfc3339_date_range_max = date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    rfc3339_date_range_min = date_before.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    return rfc3339_date_range_min, rfc3339_date_range_max


def dump_file(file, date='now', aws_env=None):
    # dump file
    if date == 'now':
        _time = datetime.now(timezone('Asia/Seoul'))
    else:
        _time = date

    if aws_env:
        dump_file_path = f"tmp/{_time.strftime('%Y-%m-%d')}.json"
    else:
        dump_file_path = f"{_time.strftime('%Y-%m-%d')}.json" 
    with open(dump_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(file, json_file, ensure_ascii=False, indent=4)


def write_file(name, file, aws_env=None):
    if aws_env:
        file_name = f'tmp/{name}'
    else:
        # file_name = name
        file_name = f'{name}'
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(file)


def remove_files(filename):
    try:
        os.remove(os.path.join("Downloads", f"{filename}.mp4"))
        os.remove(os.path.join("Downloads", f"{filename}.mp3"))
    except FileNotFoundError as e:
        print(e)
        

def remove_before_substring(s, sub):
    # sub 문자열 위치
    index = s.find(sub)

    if index != -1:
        return s[index:]
    else:
        return s


def extract_text_from_tag(data, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    content = re.findall(pattern, data, re.DOTALL)
    return content


def convert_duration(duration):
    # ISO 8601 Time format
    pattern = re.compile(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)

    # extract each time units
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0
    seconds = int(match.group(4)) if match.group(4) else 0

    return f"{hours}:{minutes:02d}:{seconds:02d}"


def is_working():
    import boto3
    
    s3 = boto3.client('s3')
    now_time = datetime.now(timezone('Asia/Seoul'))
    day_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    bucket_name = 'content-summarizer'
    folder_name = now_time.strftime("%Y-%m-%d")
    
    try:
        s3.head_object(Bucket=bucket_name, Key=f'backup/{folder_name}/')
        return True, day_dict[now_time.weekday()]
    
    except s3.exceptions.ClientError:
        return False, day_dict[now_time.weekday()]
    
    
def s3_upload(file_list):
    import boto3
    
    now_time = datetime.now(timezone('Asia/Seoul'))
    stored_name_list = list(map(lambda x: x.split("/")[-1], file_list))
    s3 = boto3.client('s3')
    bucket_name = 'content-summarizer'
    folder_name = now_time.strftime("%Y-%m-%d")
    s3.put_object(Bucket=bucket_name, Key=(f'backup/{folder_name}/'))
    for file, name in zip(file_list, stored_name_list):
        s3.upload_file(file, bucket_name, os.path.join(f'backup/{folder_name}/', name))
    
    return True


def write_s3_list(email_list, bucket_name, file_name):
    import boto3
    
    email_json = json.dumps({"LIST": email_list}, indent=4)
    s3 = boto3.client('s3')
    try:
        response = s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=email_json
        )
    except Exception as e:
        print(e)


def get_s3_list(bucket, key):
    import boto3
    
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket = bucket, Key = key)
    
    return json.loads(data['Body'].read())['LIST']


def send_verify_email(email_address_list):
    import boto3
    
    aws_region = "ap-northeast-2"
    client = boto3.client('ses', region_name=aws_region)
    try:
        for address in email_address_list:
            response = client.verify_email_address(
                EmailAddress=address
            )
    except Exception as e:
        print(e)


def send_email(recipient_list, cc_list, attachment_list, contents):
    import boto3
    from botocore.exceptions import ClientError
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    
    charset = "UTF-8"
    sender = "GenAI's <genai@lgespartner.com>"
    aws_region = "ap-northeast-2"
    now_time = datetime.now(timezone('Asia/Seoul'))
    subject =  f'[{now_time.isocalendar().year} Week {now_time.isocalendar().week}.] Youtube Trending Report' 
    client = boto3.client('ses', region_name=aws_region)
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject 
    msg['From'] = sender
    msg['To'] = ', '.join(recipient_list)
    msg['Cc'] = ', '.join(cc_list)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('src/format/email_body.html')
    date_week = now_time.isocalendar().week
    body = template.render(
        date_week = date_week, 
        contents = contents
    )
    msg_body = MIMEMultipart('alternative')
    htmlpart = MIMEText(body.encode(charset), 'html', charset)
    msg_body.attach(htmlpart)
    msg.attach(msg_body)
    verified_email_list = client.list_verified_email_addresses()['VerifiedEmailAddresses']
    recipient_list = [email for email in recipient_list if email in verified_email_list]
    cc_list = [email for email in cc_list if email in verified_email_list]
    attachment_list = sorted([item for item in attachment_list if now_time.strftime('%Y-%m-%d') in item])

    for f in attachment_list:
        path = os.path.join(os.getcwd(), f)
        with open(path, 'rb') as fil:
            msg.attach(
                MIMEApplication(
                    fil.read(),
                    Content_Disposition='attachment; filename="{}"' .format(os.path.basename(f)),
                    Name=os.path.basename(f),
                )
            )
    try:
        response = client.send_raw_email(
            Source=sender,
            Destinations=recipient_list+cc_list,
            RawMessage={
                'Data':msg.as_string(),
            },
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
    
    return True 


def extract_denser_summaries_and_missing_entities_and_original_transcript(item):
    text = " ".join([item['text'] for item in item['caption']])
    text = text.replace("[Music]", "").replace("[음악]", "")
    original_transcript = text[:500000]
    denser_summaries = []
    all_missing_entities = []
    accumulated_entities = []
    cod_entries = item.get('cod', [])
    
    for cod_item in cod_entries:
        if cod_item == None:
            continue
        if 'Denser_Summary' in cod_item:
            denser_summaries.append(cod_item['Denser_Summary'])
        if 'Missing_Entities' in cod_item:
            entities_list = cod_item['Missing_Entities'].split('; ')
            accumulated_entities.extend(entities_list)
            all_missing_entities.append(accumulated_entities.copy())
    return [denser_summaries, all_missing_entities, [original_transcript]]


def search_google_news(api_key, query="Electric Vehicle", location="United States", hl="en", gl="us"):
    # 오늘 날짜와 한달 전 날짜 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # 검색 쿼리에 날짜 범위 추가
    formatted_query = f'{query} after:{start_date.strftime("%Y-%m-%d")} before:{end_date.strftime("%Y-%m-%d")}'

    params = {
        "q": formatted_query,
        "tbm": "nws",  # News search
        "engine": "google_news",
        "location": location,
        "hl": hl,
        "gl": gl,
        "google_domain": "google.com",
        "api_key": api_key
    }

    search = serpapi.GoogleSearch(params)
    json_results = search.get_dict()
    return json_results
    

def perform_google_news_search(api_key, base_query, event_list, location="United States", hl="en", gl="us", get_n=50):
    lst_news_titles = []
    seen_links = set()
    for idx, keyword in enumerate(event_list):
        query = f"{base_query} {keyword}"
        results = {}
        try:
            results = search_google_news(api_key, query=query, location=location, hl=hl, gl=gl)
            news_results = results.get('news_results')
            for article in news_results:
                link = article.get('link')
                if link and link not in seen_links:
                    seen_links.add(link)
                    lst_news_titles.append(article)
        except Exception as e:
            # print(f"Error occurred while searching '{query}': {str(e)}")
            continue
        
    return lst_news_titles[:get_n]


def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
    ]
    return {'User-Agent': random.choice(user_agents)}


def fetch_article_info(google_news_result, headers):
    article_list = []
    for idx, article in enumerate(google_news_result):
        link = article.get('link')
        title = article.get('title')
        if not link or not title:
            # print(f"Missing data: link or title not found for an article in index {idx}")
            continue
        try:
            response = requests.get(link, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in content if p.get_text().strip()])
            article_info = {
                'title': title,
                'link': link,
                'text': article_text
            }
            article_list.append(article_info)
        except requests.Timeout:
            # print(f"Timeout occurred when retrieving: {link}")
            continue
        except requests.RequestException as e:
            # print(f"Error retrieving article: {e}")
            continue
    return article_list


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))