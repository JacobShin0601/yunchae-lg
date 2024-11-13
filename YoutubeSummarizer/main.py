import glob
import os
import argparse
import time
from datetime import datetime
import sys
from dotenv import load_dotenv
from src.utils import get_config, get_rfc_date, get_key_config, is_working
from src.thirdparty.youtube_fn import YouTubeAPI
from src.wrapper import youtube_summary_wrapper
import pandas as pd


parser = argparse.ArgumentParser(description='Summarizer')
parser.add_argument('-o', '--option', type=str, default='youtube', choices=['youtube', 'naver'],
                    help='choice selected option')
parser.add_argument('-y', '--youtube_option', type=str, default='query', choices=['query', 'channel'],
                    help='choice selected youtube search option')
parser.add_argument('-g', '--get_results', type=int, default=10)
parser.add_argument('-d', '--date_before', type=int, default=7)
parser.add_argument('-t', '--target_date', type=str, default='now')
parser.add_argument('-a', '--aws_env', type=bool, default=False)
parser.add_argument('-c', '--num_of_comments', type=int, default=3)
parser.add_argument('-w', '--day_of_the_week', type=str, default='Mon')


if __name__ == "__main__":
    start = time.time()
    load_dotenv()
    config = get_config()
    args = parser.parse_args()
    
    if args.target_date == 'now':
        target_date = 'now'
        [publishedAfter, publishedBefore]=get_rfc_date(before_days=args.date_before)
    else:
        target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
        [publishedAfter, publishedBefore]=get_rfc_date(date=target_date, before_days=args.date_before)

    if args.aws_env:
        key_config = get_key_config()
        folder_check, day_check = is_working()
        # if (args.day_of_the_week != day_check) | (folder_check==True):
        #     print("Stops the function because it is a day of the week that you have already send or are not sending.")
        #     sys.exit()
    else:
        key_config = {}
        key_config['Anthropic'] = {'key':os.environ.get('ANTHROPIC_API')}
        key_config['youtube_api'] = {'key':os.environ.get('YOUTUBE_API')}
        key_config['SERP_API'] = {'key':os.environ.get('SERP_API')}

    if args.option == 'youtube':
        youtube = YouTubeAPI(
            api_key = key_config['youtube_api']['key'],
            service_name="youtube",
            version="v3",
            publishedAfter=publishedAfter,
            publishedBefore=publishedBefore
        )

        ys = youtube_summary_wrapper(youtube, args, config, key_config)
        ys.run(target_date)
        
    end = time.time()
    print(f"Time Elapsed: {end - start} seconds")