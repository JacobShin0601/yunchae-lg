import pandas as pd
import glob
import os
import sys
import argparse
from src.utils import s3_upload, get_s3_list, send_email, get_config, get_key_config, is_working

parser = argparse.ArgumentParser(description='Summarizer')
parser.add_argument('-w', '--day_of_the_week', type=str, default='Mon')

if __name__ == "__main__":
    config = get_config()
    args = parser.parse_args()
    key_config = get_key_config()
    folder_check, day_check = is_working()
    if (args.day_of_the_week != day_check) | (folder_check==True):
        print("Stops the function because it is a day of the week that you have already send or are not sending.")
        sys.exit()
    recipient_list = get_s3_list(
        bucket = 'content-summarizer', 
        key = 'recipient.json'
    )
    cc_list = get_s3_list(
        bucket = 'content-summarizer', 
        key = 'cc.json'
    )
    file_list= glob.glob("tmp/*")
    attachment_list = [i for i in file_list if '.html' in i]
    s3_upload(file_list)
    contents = [
        "사용자 접근성을 고려한 UI 개선을 했습니다.",
        "콘텐츠 요약 대상을 주요 OEM, 원재료, 경쟁사, 외부환경으로 확장했습니다.",
        "요약을 수행하는 생성형 AI모델을 최신 버전으로 업데이트 했습니다.(Claude3 -> Claude3.5 Sonnet)"
    ] 

    for email in recipient_list:
        send_email([email], cc_list, attachment_list, contents)
    for file in file_list:
        os.remove(file)