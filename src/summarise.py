import openai
from openai import AsyncOpenAI, OpenAI
import os
from src.utils import load_env, load_config


class Summary():
    def __init__(self):
        # load env
        env_vars = load_env()
        self.LLM_API_KEY = env_vars['LLM_API_KEY']
        # load config
        self.config = load_config(config_path="config.yaml")

    def initialise_openai(self):
        # initialise the OpenAI API
        openai.api_key = self.LLM_API_KEY
        self.OAI_CLIENT = OpenAI(
            api_key=self.LLM_API_KEY, max_retries=2)

    def Summarise(self, newsString):
        res = self.OAI_CLIENT.chat.completions.create(
            model=self.config['model'],
            n=1,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a journalist."},
                {"role": "user", "content": "The news of the day is as follows:"},
                {"role": "user",
                 "content": newsString},
                {"role": "user", "content": "Summarise the news in succinct terms."},
            ],
            stream=False
        )
        return res.choices[0].message.content, res
