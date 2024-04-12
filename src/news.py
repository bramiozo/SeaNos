# this module pulls in news from a news API
# the news API is specified in the config file

from newsapi import NewsApiClient
from datetime import datetime, timedelta


class News:
    # https://newsapi.org/docs/endpoints/top-headlines
    def __init__(self, api_key):
        self.api_key = api_key
        self.NewsAPI = NewsApiClient(api_key=self.api_key)

    @staticmethod
    def _collector(headlines, first_n=5):
        concatted = ""
        for headline in headlines['articles'][:first_n]:
            concatted += "Title:" + headline['title'] + ". "
            if headline['description'] is not None:
                concatted += "Description:" + headline['description'] + ". "
            if headline['content'] is not None:
                concatted += "Content:" + headline['content'] + ". "
            concatted += "------------------------------"
        return concatted

    def get_top_news(self, query):
        # get the news
        headlines = self.NewsAPI.get_top_headlines(q=query,
                                                   # sources='bbc-news,the-verge',
                                                   language='en')
        res = f"Here are the latest news: {self._collector(headlines)}"
        return res

    def get_any_news(self, query):
        # from 7 days ago
        from_date = (datetime.now()-timedelta(days=7)).date().isoformat()

        # get the news
        headlines = self.NewsAPI.get_everything(q=query,
                                                from_param=from_date,
                                                # sources='bbc-news,the-verge',
                                                sort_by='popularity',
                                                language='en')
        res = f"Here are the latest news: {self._collector(headlines)}"
        return res
