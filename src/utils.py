import dotenv
from dotenv import find_dotenv
import os
import benedict
import pandas as pd
import argparse

def load_env():
    # load the .env file
    dotenv.load_dotenv(dotenv_path=find_dotenv())

    # get the environment variables
    return {
        "LLM_API_KEY": os.getenv("LLM_API_KEY"),
        "LLM_API_SECRET": os.getenv("LLM_API_SECRET"),
        "LLM_API_URL": os.getenv("LLM_API_URL"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
    }


def load_config(config_path: str):
    # load the YAML configuration file
    config_dict = benedict.benedict(config_path, format="yaml")
    return config_dict
