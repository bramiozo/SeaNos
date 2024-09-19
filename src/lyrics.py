# a module that encodes the functionality to approach an LLM API such as ChatGPT
# to generate lyrics for a song based on a given prompt
# The prompts and overall settings are loaded from the config file

# https://platform.openai.com/docs/guides/text-to-speech

import openai
# import asyncio
from openai import AsyncOpenAI, OpenAI
# import datetime
# import os
import random

from src.utils import load_env

class Translate:
    def __init__(self, config):
        self.config = config
        self.target_language = config['target_language']
        self.system_prompt = config["prompts"]["system_translation"]
        self.temperature = config["temperature"]
        self.num_responses = config["num_responses"]
        self.model = config["model"]
        self.keys = load_env()

    def initialise_openai(self):
        # initialise the OpenAI API
        openai.api_key = self.keys['LLM_API_KEY']
        self.OAI_ASYNC_CLIENT = AsyncOpenAI(
            api_key=self.keys['LLM_API_KEY'], max_retries=2)
        self.OAI_CLIENT = OpenAI(
            api_key=self.keys['LLM_API_KEY'], max_retries=2)

    def run_model(self, GaelicSong):
        # load the model
        # openai.ChatCompletion.create
        return self.OAI_CLIENT.chat.completions.create(
            model=self.model,
            n=self.num_responses,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "The Gaelic song is as follows:"},
                {"role": "user",
                 "content": GaelicSong},
                {"role": "user", "content": f"Generate the {self.target_language} translation of the song"},
            ],
            stream=False
        )

    def generate(self, GaelicSong=None):
        # generate the lyrics for the song
        res = self.run_model(GaelicSong=GaelicSong)
        # return the generated lyrics
        return res.choices[0].message.content, res


class Lyrics:
    def __init__(self, config):
        self.config = config
        self.system_prompt = config["prompts"]["system"]
        self.user_prompts = config["prompts"]["user"]
        self.temperature = config["temperature"]
        self.num_responses = config["num_responses"]
        self.model = config["model"]
        self.keys = load_env()

    def initialise_openai(self):
        # initialise the OpenAI API
        openai.api_key = self.keys['LLM_API_KEY']
        self.OAI_ASYNC_CLIENT = AsyncOpenAI(
            api_key=self.keys['LLM_API_KEY'], max_retries=2)
        self.OAI_CLIENT = OpenAI(
            api_key=self.keys['LLM_API_KEY'], max_retries=2)

    def run_model(self, newsContent):
        # load the model
        # openai.ChatCompletion.create
        max_len = len(self.user_prompts)
        rnd_sel = random.randint(0, max_len - 1)
        return self.OAI_CLIENT.chat.completions.create(
            model=self.model,
            n=self.num_responses,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompts[rnd_sel]},
                {"role": "user", "content": "The news is as follows:"},
                {"role": "user",
                 "content": newsContent},
                {"role": "user", "content": "Generate the lyrics for the song"},
            ],
            stream=False
        )

    def generate(self, newsContent=None):
        # generate the lyrics for the song
        res = self.run_model(newsContent=newsContent)
        # return the generated lyrics
        return res.choices[0].message.content, res
