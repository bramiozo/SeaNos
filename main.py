# loads config with the settings for the lyrics generation
#
from src.lyrics import Lyrics
from src.news import NewsAPI, RSSParser
from src.speak import Speak
from src.utils import load_config, load_env
from src.summarise import Summary

import os
import sys
import argparse
import logging
from time import sleep
from datetime import datetime
import random

# TODO: add logger
# TODO: put the generate function in a loop?

# initialise the logger
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler("lyrics.log")
fh.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(fh)


# TODO: new News classes, output list of dictionaries,
# TODO: output to sqlite db
# TODO: make sqlite db class to feed the news/lyrics

def lyrics(query=None,
           debug=False,
           topNews=False,
           newsSource='RSS',
           summary=False,
           newsSelection='random',
           configpath="config.yaml"):
    logger.info("Starting the lyrics generation process")
    logger.info(datetime.now())

    lyricist = Lyrics(config=load_config(config_path=configpath))
    lyricist.initialise_openai()
    print("Query:", query)
    if query is not None:
        lyricist.news_topic = query

    keys = load_env()
    if newsSource == 'newsapi':
        NewsGetter = NewsAPI(api_key=keys['NEWS_API_KEY'])
        if topNews == True:
            newsList = NewsGetter.get_top_news(query)
        else:
            newsList = NewsGetter.get_any_news(query)
    elif newsSource == 'RSS':
        RSSGetter = RSSParser(config_path=configpath)
        newsList = RSSGetter.parse_feeds()

    if len(newsList) == 0:
        logger.error(f"No news found for q:{query}")
        newsString = "Please be kind to us."

    if newsSelection == 'random':
        d = random.choice(newsList)
        newsString = f'Title:{d["title"]},Description:{d["description"]}'
    elif newsSelection == 'all':
        newsString = " ".join(
            [f'Title:{d["title"]},Description:{d["description"]}'
             for d in newsList])
    elif newsSelection.isdigit():
        newsString = " ".join(
            [f'Title:{d["title"]},Description:{d["description"]}'
             for d in newsList[:int(newsSelection)]])
    else:
        raise ValueError("newsSelection must be 'random', 'all' or an integer")

    if summary:
        # Use OpenAI to summarise the news
        srizer = Summary()
        srizer.initialise_openai()
        newsString, OAI_summary = srizer.Summarise(newsString)
        if debug:
            logger.info(10*"<" + "DEBUGGING SUMMARY" + 10*">")
            logger.info(str(OAI_summary))
            logger.info(10*"<" + "DEBUGGING" + 10*">")

    if debug:
        logger.info(10*"<" + "DEBUGGING newsString" + 10*">")
        logger.info(str(newsString))
        logger.info(10*"<" + "DEBUGGING" + 10*">")

    lyrics, OAI_lyrics = lyricist.generate(newsString)

    if debug:
        logger.info(10*"<" + "DEBUGGING LYRICS" + 10*">")
        logger.info(str(OAI_lyrics))
        logger.info(10*"<" + "DEBUGGING" + 10*">")

    return lyrics


def speak(lyrics):
    '''
    tts = TTS(config=load_config(config_path="config.yaml"))
    tts.generate(lyrics)
    '''
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Headline news",
                        help="Query to generate lyrics for")
    parser.add_argument("--debug", type=bool, help="Debug mode", default=False)
    parser.add_argument("--refresh", type=int, help="Refresh rate in seconds",
                        default=None)
    parser.add_argument("--topNews", type=bool,
                        help="Get top news", default=False)
    parser.add_argument("--newsSource", type=str,
                        help="News source", default='RSS')
    parser.add_argument("--newsSelection", type=str,
                        help="News selection", default='random')
    parser.add_argument("--summary", type=bool,
                        help="Summarise the news", default=False)

    args = parser.parse_args()
    query = args.query
    debug = args.debug
    refresh_rate = args.refresh
    topNews = args.topNews
    newsSource = args.newsSource
    newsSelection = args.newsSelection
    summary = args.summary

    # static, one off run
    lyrics_text = lyrics(query=query,
                         debug=debug,
                         topNews=topNews,
                         newsSource=newsSource,
                         newsSelection=newsSelection,
                         summary=summary,
                         configpath="config_demo.yaml")

    # text to phonetics
    phonetics = Phonetics(lyrics_text, source_lang="en", target_lang="gle")

    # language: gle (Irish)
    speaker = Speak(language="en", config_path="config_demo.yaml")
    speaker.speak(lyrics_text, OutPath="./artifacts/output.wav")
