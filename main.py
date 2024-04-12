# loads config with the settings for the lyrics generation
#
from src.lyrics import Lyrics
from src.news import News
from src.tts import TTS
from src.utils import load_config

import os
import sys
import argparse


def main(query=None, debug=False):
    lyricist = Lyrics(config=load_config(config_path="config.yaml"))
    lyricist.initialise_openai()
    if query is not None:
        lyricist.news_topic = query

    lyrics, complete = lyricist.generate()
    if debug:
        print(30*"#", "DEBUGGING", 30*"#")
        print(complete)
        print(30*"#", "DEBUGGING", 30*"#")
    return lyrics

    '''
    tts = TTS(config=load_config(config_path="config.yaml"))
    tts.generate(lyrics)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Headline news",
                        help="Query to generate lyrics for")
    parser.add_argument("--debug", type=bool, help="Debug mode", default=False)
    query = parser.parse_args().query
    debug = parser.parse_args().debug
    main(query=query, debug=debug)
