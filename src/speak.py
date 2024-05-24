import torch
import transformers
from TTS.api import TTS
from TTS.tts.models.vits import Vits
from TTS.tts.models.xtts import Xtts
from src.utils import load_config
import os
# here is the module to generate the speech wav, based on the lyrics
# the lyrics are generated using the lyrics module

# https://github.com/coqui-ai/TTS
# https://github.com/facebookresearch/fairseq/tree/main/examples/mms


# models : ylacombe/vits_vctk_irish_male, ylacombe/vits_ljs_irish_male
# models: neongeckocom/tts-vits-cv-ga
# models: https://aimodels.org/ai-models/text-to-speech-synthesis/irish-female-tts-model-vits-encoding-trained-on-cv-dataset-at-22050hz/
# transformers.pipeline('text-to-speech', model='ylacombe/vits_vctk_irish_male')

class Speak:
    def __init__(self,
                 config_path: str = "config.yaml",
                 language: str = "gle",  # ga
                 model_path: str = None):
        """
        Initialize the Speak class.

        Args:
            config_path (str): Path to the TTS config file.
            language (str): Language code.
            model_path (str): Path to the TTS model.
        """
        self.config = load_config(config_path)
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = self.config['tts']['model_path']

        config_path = self.config['tts'].get('config_path')
        if config_path is None:
            config_path = os.path.join(self.model_path, "config.json")

        self.tts = TTS(progress_bar=True,
                       model_path=self.model_path,
                       config_path=config_path)
        self.tts.to('gpu')

        # state_dict = torch.load(self.model_path)["model"]
        # self.tts.tts.load_state_dict(state_dict, strict=False)

        self.lang = language

    def speak(self, lyrics, OutPath):
        """
        Generate speech from given lyrics and save to file.

        Args:
            lyrics (str): Text to convert to speech.
            out_path (str): Path to save the output speech file.
        """
        self.tts.tts_to_file(
            text=lyrics,
            # speaker=self.tts.speakers[0],
            language=self.lang,
            speaker_wav="assets/english_bram.wav",
            file_path=OutPath
        )
        pass

    def convert(self):
        """
        Convert the voice of a speaker to another speaker.

        Args:
            source_wav (str): Path to the source wav file.
            target_wav (str): Path to the target wav file.
            file_path (str): Path to save the output wav file.
        """
        self.tts.voice_conversion_to_file(
            source_wav="assets/sweet_scarlet.wav",
            target_wav="assets/voice_conversion_target.wav",
            file_path="assets/voice_conversion_output.wav"
        )
        pass

    def clone(self, Text, SpeakerPath, OutPath):
        """
        Generate speech with voice conversion.

        Args:
            text (str): Text to convert to speech.
            speaker_path (str): Path to the speaker file.
            out_path (str): Path to save the output speech file.
        """
        self.tts.tts_with_vc_to_file(text=Text,
                                     language=self.lang,
                                     speaker_wav=SpeakerPath,
                                     file_path=OutPath
                                     )
        pass


if __name__ == "__main__":
    # load the model
    tts = TTS()
    print(tts.list_models())
