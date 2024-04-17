# loads config with the settings for the lyrics generation
#
from src.lyrics import Lyrics
from src.news import NewsAPI
from src.tts import TTS
from src.utils import load_config, load_env

import os
import sys
import argparse
import logging
from time import sleep
from datetime import datetime

# TODO: add logger
# TODO: put the generate function in a loop?

# initialise the logger
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler("lyrics.log")
fh.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(fh)


# TODO: new News classes, output list of dictionaries,
# lyrics/lyrics_generator functions need minor refactoring.

def lyrics(query=None, debug=False, topNews=False):
    logger.info("Starting the lyrics generation process")
    logger.info(datetime.now())

    lyricist = Lyrics(config=load_config(config_path="config.yaml"))
    lyricist.initialise_openai()
    print("Query:", query)
    if query is not None:
        lyricist.news_topic = query

    keys = load_env()
    NewsGetter = NewsAPI(api_key=keys['NEWS_API_KEY'])

    if topNews == True:
        newsString = NewsGetter.get_top_news(query)
    else:
        newsString = NewsGetter.get_any_news(query)

    if newsString == "":
        logger.error(f"No news found for q:{query}")
        newsString = "There shall be peace in our times."
    lyrics, complete = lyricist.generate(newsString)

    if debug:
        logger.info(10*"<" + "DEBUGGING" + 10*">")
        logger.info("NEWS STRING")
        logger.info(newsString)
        logger.info("LYRICS")
        logger.info(str(complete))
        logger.info(10*"<" + "DEBUGGING" + 10*">")

    return lyrics


def lyrics_generator(query=None, debug=False, refresh_rate=None):
    logger.info("Starting the lyrics generation iterator")
    logger.info(datetime.now())

    # loop with refresh rate of refresh_rate seconds
    # use yield to return the lyrics
    lyricist = Lyrics(config=load_config(config_path="config.yaml"))
    lyricist.initialise_openai()
    print("Query:", query)
    if query is not None:
        lyricist.news_topic = query

    keys = load_env()
    NewsGetter = NewsAPI(api_key=keys['NEWS_API_KEY'])

    counter = 0
    while True:
        newsString = NewsGetter.get_top_news(query)
        if newsString == "":
            logger.error(f"No news found for q:{query}")
            newsString = "There shall be peace in our times."
        lyrics, complete = lyricist.generate(newsString)

        sleep(refresh_rate)

        if debug:
            logger.info(10*"<" + "DEBUGGING" + 10*">")
            logger.info(datetime.now())
            logger.info("NEWS STRING")
            logger.info(newsString)
            logger.info("LYRICS")
            logger.info(str(complete))
            logger.info(10*"<" + "DEBUGGING" + 10*">")
            break
        yield lyrics


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
                        help="Get top news", default=True)

    args = parser.parse_args()
    query = args.query
    debug = args.debug
    refresh_rate = args.refresh
    topNews = args.topNews

    # static, one off run
    lyrics(query=query, debug=debug, topNews=topNews)

    # loop, iterator
    # main(query=query, debug=debug, refresh_rate=refresh_rate)
