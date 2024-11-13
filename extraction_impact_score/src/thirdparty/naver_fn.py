import os
import urllib.request
import json
import requests

from langchain.document_loaders import WebBaseLoader

class NaverAPI:
	def __init__(self, client_id:str, client_secret:str):
		self.client_id = client_id
		self.client_secret = client_secret


	def search(self, quote:str, num:int=10):
		encText = urllib.parse.quote(quote)

		url = "https://openapi.naver.com/v1/search/news.json?query=" + encText + "&display=" + str(num) + "&sort=date"# json 결과

		request = urllib.request.Request(url)
		request.add_header("X-Naver-Client-Id", self.client_id)
		request.add_header("X-Naver-Client-Secret", self.client_secret)

		response = urllib.request.urlopen(request)
		rescode = response.getcode()

		if (rescode == 200):
			response_body = response.read()
			response_data = response_body.decode('utf-8')
			
			return response_data
			
		else:
			print("Error Code : " + rescode)

	
	def get_news_infos(self, args, config):
		news_dict = {}
		for query in config['Query']['naver']:
			news_dict[query] = []
			search_response = self.search(
				quote=query,
				num=args.get_results)
			news_dict[query] = json.loads(search_response)
			for news in news_dict[query]['items']:
				news['contents'] = self.get_news_content(news['link'])

		return news_dict
	

	def get_news_content(self, url):
		if 'news.naver.com' not in url:
			return '{} is not requestable url'.format(url)
		
		res = requests.get(url, verify=False)
		
		if res.status_code == 200:
			soup = BeautifulSoup(res.text)
			article = soup.find('div', "newsct_article").text.strip()
			# evade flash error
			article = article.replace('// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}', '')
			#content = soup.select_one('#articleBodyContents')

			return article
		else:
			raise Exception('Not 200 status code')