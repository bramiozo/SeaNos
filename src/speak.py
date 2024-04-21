import torch
from TTS.api import TTS
from src.utils import load_config

# https://github.com/coqui-ai/TTS
# https://github.com/facebookresearch/fairseq/tree/main/examples/mms

class Speak:
    def __init__(self,
                 config_path="../config.yaml",
                 language="gle",
                 model_path=None):
        self.config = load_config(config_path)
        self.tts = TTS(progress_bar=True, model_path=model_path, gpu=True)
        self.lang = language

    def speak(self, lyrics, OutPath):
        self.tts.tts_to_file(
            text=lyrics,
            language=self.lang,
            speaker_wav="../assets/sweet_scarlet.wav",
            file_path=OutPath
        )
        pass

    def convert(self):
        self.tts.voice_conversion_to_file(
            source_wav="../assets/sweet_scarlet.wav",
            target_wav="../assets/voice_conversion_target.wav",
            file_path="../assets/voice_conversion_output.wav"
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
