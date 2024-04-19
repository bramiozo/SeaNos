# this module pulls in news from a news API
# the news API is specified in the config file

from newsapi import NewsApiClient
from datetime import datetime, timedelta
from src.utils import load_config, load_env
import feedparser

# https://newsdata.io/pricing
# https://newsapi.org/pricing
# https://worldnewsapi.com/pricing/


class NewsAPI:
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

    @staticmethod
    def _concatenate(headlines):
        # concat 'content' and 'description' fields
        headlines = [{'title': headline['title'],
                      'description': headline.get('description')+"|" +
                      headline.get('content')
                      } for headline in headlines]
        return headlines

    def get_top_news(self, query):
        # get the news
        headlines = self.NewsAPI.get_top_headlines(q=query,
                                                   # sources='bbc-news,the-verge',
                                                   language='en')
        return self._concatenate(headlines['articles'])

    def get_any_news(self, query):
        # from 7 days ago
        from_date = (datetime.now()-timedelta(days=7)).date().isoformat()

        # get the news
        headlines = self.NewsAPI.get_everything(q=query,
                                                from_param=from_date,
                                                # sources='bbc-news,the-verge',
                                                sort_by='popularity',
                                                language='en')
        return self._concatenate(headlines['articles'])


class RSSParser:
    def __init__(self, sources=None, config_path="../config.yaml"):
        config = load_config(config_path)
        DEFAULT_RSS_SOURCES = config['rss_sources']

        if isinstance(sources, list):
            self.sources = sources
        else:
            self.sources = DEFAULT_RSS_SOURCES

    def parse_feeds(self):
        news_items = []
        for _, feedData in self.sources.items():
            parsed_feed = feedparser.parse(feedData['url'])
            for entry in parsed_feed.entries:
                news_items.append({
                    'title': entry[feedData['title']],
                    'description': entry[feedData['description']],
                })
        return news_items


if __name__ == "__main__":
    # test the NewsAPI
    keys = load_env()
    NewsGetter = NewsAPI(api_key=keys['NEWS_API_KEY'])

    print(30*"-")
    print(NewsGetter.get_top_news('Ireland')[:3])
    print(NewsGetter.get_any_news('Ireland')[:3])
    print(30*"-")

    # test RSS Parser
    rss_parser = RSSParser()
    news_items = rss_parser.parse_feeds()
    print(news_items[:3])
