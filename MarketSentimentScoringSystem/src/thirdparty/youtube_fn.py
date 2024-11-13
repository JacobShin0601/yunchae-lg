import os
import gc
import markdown

from googleapiclient.discovery import build
from typing import Any
from youtube_transcript_api import YouTubeTranscriptApi

from ..utils import get_rfc_date, write_file, convert_duration


class YouTubeAPI:
    def __init__(self, api_key:str, service_name:str='youtube', version:str='v3', publishedAfter:Any=None, publishedBefore:Any=None):
        self.youtube = build(service_name, version, developerKey=api_key)
        self.publishedAfter = publishedAfter
        self.publishedBefore = publishedBefore

    def search(self, q:str=None, order:str = "relevance", channel_id:str=None, maxResults:int=None) -> Any:
        """search _summary_

        Args:
            q (str, optional): _description_. Defaults to None.
            order (str, optional): _description_. Defaults to "rating".
            channel_id (str, optional): _description_. Defaults to None.
            publishedAfter (int, optional): _description_. Defaults to 15.
            maxResults (int, optional): _description_. Defaults to None.

        Returns:
            Any: _description_
        """
        search_response = self.youtube.search().list(
            q = q,
            order = order,
            channelId = channel_id,
            publishedAfter = self.publishedAfter,
            publishedBefore = self.publishedBefore,
            part = 'snippet',
            type = 'video',
            videoDefinition = 'high',
            videoLicense = 'youtube',
            maxResults = maxResults
            ).execute()
        
        return search_response
    
    def get_video_infos(self, args, config) -> dict:
        """get_video_infos _summary_

        Args:
            args (_type_): _description_
            config (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            dict: _description_
        """
        self.video_dict = {}
        # print('args in get_video_infos of youtube_fn.py:', args)
        # print('args.youtube_option in get_video_infos of youtube_fn.py:', args['youtube_option'])
        if args['youtube_option'] == 'channel':
            for channel_id in config['Channel']['channel_id']:
                for keyword in config['Channel']['keywords']:                
                    self.video_dict[channel_id] = []
                    search_response = self.search(
                        q=keyword,
                        order='relevance', 
                        channel_id=channel_id, 
                        maxResults=args['get_results'])
                    self.video_dict[channel_id] += search_response['items']

        elif args['youtube_option'] == 'query':
            for query in config['Query']['querys']:
                self.video_dict[query] = []
                search_response = self.search(
                    q=query,
                    order='relevance',
                    maxResults=args['get_results'])
                self.video_dict[query] += search_response['items']

            '''
            ## remove duplicated videos
            videos_id = []
            for query in self.video_dict.keys():
                for videos in self.video_dict[query]:
                    if videos['id']['videoId'] in videos_id:
                        videos['duplicated'] = True
                    else:
                        videos['duplicated'] = False
                        videos_id.append(videos['id']['videoId'])

            del videos_id
            '''
            
        else:
            raise ValueError("option is not valid")
        
        return self.video_dict
    
    def get_video_duration(self, video_id:str):
        video_response = self.youtube.videos().list(
            id=video_id,
            part='contentDetails'
        ).execute()

        content_details = video_response['items'][0]['contentDetails']
        duration = content_details['duration']
        converted_duration = convert_duration(duration)
        converted_duration = ':'.join([t+u for t,u in zip(converted_duration.split(':'), ['h','m','s'])])

        return content_details, converted_duration

    def get_caption(self, video_id:str):
        """get_caption _summary_

        Args:
            video_id (str): _description_

        Returns:
            _type_: _description_
        """
        # assigning srt variable with the list 
        # of dictionaries obtained by the get_transcript() function
        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ko'])
        except:
            #print("subtitle is not available")
            srt = []
        
        # return the result
        return srt

    def get_statistics(self, video_id):
        video_response = self.youtube.videos().list(
            part='snippet, statistics', 
            id=video_id,
        ).execute()['items'][0]
        statistics_dict = dict()
        statistics_dict['viewCount'] = video_response['statistics']['viewCount'] if 'viewCount' in video_response['statistics'] else 0
        statistics_dict['likeCount'] = video_response['statistics']['likeCount'] if 'likeCount' in video_response['statistics'] else 0
        statistics_dict['commentCount'] = video_response['statistics']['commentCount'] if 'commentCount' in video_response['statistics'] else 0
        
        return statistics_dict
    
    def get_comments(self, video_id, order='relevance', max_result=100, all_result=False):
        comments = list()
        try:
            comment_response = self.youtube.commentThreads().list(
                part='snippet,replies', 
                videoId=video_id, 
                order=order,
                maxResults=max_result
            ).execute()

            if all_result == True:
                while comment_response:
                    for item in comment_response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        comment_dict = dict()
                        comment_dict['textOriginal'] = comment['textOriginal']
                        comment_dict['authorDisplayName'] = comment['authorDisplayName']
                        comment_dict['publishedAt'] = comment['publishedAt']
                        comment_dict['likeCount'] = comment['likeCount']
                        comments.append(comment_dict)
                        if item['snippet']['totalReplyCount'] > 0:
                            for reply_item in item['replies']['comments']:
                                reply = reply_item['snippet']
                                comment_dict = dict()
                                comment_dict['textOriginal'] = comment['textOriginal']
                                comment_dict['authorDisplayName'] = comment['authorDisplayName']
                                comment_dict['publishedAt'] = comment['publishedAt']
                                comment_dict['likeCount'] = comment['likeCount']
                                comments.append(comment_dict) 
                    if 'nextPageToken' in comment_response:
                        comment_response = self.youtube.commentThreads().list(
                            part='snippet,replies', 
                            videoId=video_id,
                            order=order,
                            pageToken=comment_response['nextPageToken'],
                            maxResults=max_result
                        ).execute()
                    else:
                        break
            else:
                for item in comment_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comment_dict = dict()
                    comment_dict['textOriginal'] = comment['textOriginal']
                    comment_dict['authorDisplayName'] = comment['authorDisplayName']
                    comment_dict['publishedAt'] = comment['publishedAt']
                    comment_dict['likeCount'] = comment['likeCount']
                    comments.append(comment_dict)
        except:
            comment_dict = dict()
            comment_dict['textOriginal'] = "None"
            comment_dict['authorDisplayName'] = "None"
            comment_dict['publishedAt'] = "None"
            comment_dict['likeCount'] = "None"
            comments.append(comment_dict)
            
        return comments