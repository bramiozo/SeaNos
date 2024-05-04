import torch
import transformers
from TTS.api import TTS
from src.utils import load_config
# here is the module to generate the speech wav, based on the lyrics
# the lyrics are generated using the lyrics module

# https://github.com/coqui-ai/TTS
# https://github.com/facebookresearch/fairseq/tree/main/examples/mms


# models : ylacombe/vits_vctk_irish_male, ylacombe/vits_ljs_irish_male
# transformers.pipeline('text-to-speech', model='ylacombe/vits_vctk_irish_male')

class Speak:
    def __init__(self,
                 config_path="config.yaml",
                 language="gle",
                 model_path=None):
        self.config = load_config(config_path)
        self.tts = TTS(progress_bar=True,
                       model_path=self.config['tts']['model_path'],
                       gpu=True)
        self.lang = language
        # print("Available speakers:")
        # print(self.tts.speakers)
        # print("Available languages:")
        # print(self.tts.languages)

    def speak(self, lyrics, OutPath):
        self.tts.tts_to_file(
            text=lyrics,
            # speaker=self.tts.speakers[0],
            # language=self.lang,
            speaker_wav="assets/sweet_scarlet.wav",
            file_path=OutPath
        )
        pass

    def convert(self):
        self.tts.voice_conversion_to_file(
            source_wav="assets/sweet_scarlet.wav",
            target_wav="assets/voice_conversion_target.wav",
            file_path="assets/voice_conversion_output.wav"
        )
        pass

    def clone(self, Text, SpeakerPath, OutPath):
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
