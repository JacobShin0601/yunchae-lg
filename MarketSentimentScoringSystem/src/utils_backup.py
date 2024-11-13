import os
import glob
import platform
import json
import re
from datetime import datetime, date, timedelta
from jinja2 import Environment, FileSystemLoader
from pytz import timezone


def get_config():
    # Read config file
    config_path = "config.json"
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)

    return config


def get_key_config():
    import boto3
    
    # Read config file
    bucket = 'metal-content-summarizer'
    key = 'key.json'
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket = bucket, Key = key)
    config = json.loads(data['Body'].read())

    return config


def get_rfc_date(before_days:int=15):
    # within one month
    now_time = datetime.now(timezone('Asia/Seoul'))
    one_month_ago = now_time.today() - timedelta(before_days)
    # within 6 month
    #one_month_ago = date.today() - timedelta(180)
    rfc3339_date = one_month_ago.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    return rfc3339_date


def dump_file(file, aws_env=None):
    # dump file
    now_time = datetime.now(timezone('Asia/Seoul'))
    if aws_env:
        dump_file_path = f"/tmp/{now_time.strftime('%Y-%m-%d')}.json"
    else:
        dump_file_path = f"{now_time.strftime('%Y-%m-%d')}.json" 
    with open(dump_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(file, json_file, ensure_ascii=False, indent=4)


def write_file(name, file, aws_env=None):
    if aws_env:
        file_name = f'/tmp/{name}'
    else:
        file_name = name
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


def convert_duration(duration):
    # ISO 8601 Time format
    pattern = re.compile(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)

    # extract each time units
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0
    seconds = int(match.group(4)) if match.group(4) else 0

    return f"{hours}:{minutes:02d}:{seconds:02d}"


def s3_upload(file_list):
    import boto3
    
    now_time = datetime.now(timezone('Asia/Seoul'))
    stored_name_list = list(map(lambda x: x.split("/")[-1], file_list))
    s3 = boto3.client('s3')
    bucket_name = 'metal-content-summarizer'
    folder_name = now_time.strftime("%Y-%m-%d")
    s3.put_object(Bucket=bucket_name, Key=(f'backup/{folder_name}/'))
    for file, name in zip(file_list, stored_name_list):
        s3.upload_file(file, bucket_name, os.path.join(f'backup/{folder_name}/', name))
    
    return True


def read_recipient(bucket, key):
    import boto3
    
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket = bucket, Key = key)
    
    return json.loads(data['Body'].read())['RECIPIENT_LIST']


def send_email(recipient_list, attachment_list):
    import boto3
    import os
    from botocore.exceptions import ClientError
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    
    charset = "UTF-8"
    sender = "Yunchae Shin <yunchae@lgensol.com>"
    aws_region = "ap-northeast-2"
    now_time = datetime.now(timezone('Asia/Seoul'))
    subject =  f'[{now_time.isocalendar().year} Week {now_time.isocalendar().week}.] Market Trend Report from Youtube (Beta Test)' 
    client = boto3.client('ses', region_name=aws_region)
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject 
    msg['From'] = sender
    msg['To'] = ', '.join(recipient_list)
    # body = '안녕하세요.<br>'\
    #        '이번 주 최신 Youtube 영상을 요약한 Market Trend Report 파일을 첨부합니다.<br><br>'\
    #        '이 리포트는 AI를 활용해 중립적인 시각에서 작성되었으며 '\
    #        '배터리 소재 시장 동향을 이해하는데 도움이 되리라 확신합니다.<br>'\
    #        '추가 질문이나 개선요청사항이 있을 시 회신 부탁드립니다.<br><br>'\
    #        '감사합니다.'
    html_file_path = os.path.join(os.getcwd(), attachment_list[0])
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        body = html_file.read()          
    msg_body = MIMEMultipart('alternative')
    htmlpart = MIMEText(body.encode(charset), 'html', charset)
    msg_body.attach(htmlpart)
    msg.attach(msg_body)
    verified_email_list = client.list_verified_email_addresses()['VerifiedEmailAddresses']
    recipient_list = [email for email in recipient_list if email in verified_email_list]
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
            Destinations=recipient_list,
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