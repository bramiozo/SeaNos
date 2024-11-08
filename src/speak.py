import torch
import transformers
from TTS.api import TTS
from TTS.tts.models.vits import Vits
from TTS.tts.models.xtts import Xtts
from neon_tts_plugin_coqui import CoquiTTS as neonTTS
from IPython.display import Audio
from scipy.io import wavfile
import numpy as np
import random

from pydub import AudioSegment

import noisereduce as nr

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
                 denoise: bool = True,
                 use_neon: bool = True,
                 model_path: str = None,
                 denoise_kwargs: dict = None):
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

        if (language in ['ga', 'gle']) and (use_neon):
            self.tts = neonTTS(lang="ga", config={})
        else:
            self.tts = TTS(progress_bar=True,
                        model_path=self.model_path,
                        config_path=config_path)
            self.tts.to('cuda')

        # state_dict = torch.load(self.model_path)["model"]
        # self.tts.tts.load_state_dict(state_dict, strict=False)

        self.lang = language
        self.use_neon = use_neon
        self.denoise = denoise
        if denoise_kwargs is None:
            # load in from config
            self.denoise_kwargs =  self.config.get('denoise', {})
        else:
            if isinstance(denoise_kwargs, dict):
                self.denoise_kwargs = denoise_kwargs

    @staticmethod
    def denoiser(x: np.ndarray,
                sr: int=44_000,
                n_fft: int=2048,
                win_length: int=2048,
                hop_length: int=512,
                prop_decrease: float=0.8) -> np.ndarray:

        reduced_noise = nr.reduce_noise(y=x,
                                        sr=sr,
                                        prop_decrease=prop_decrease,
                                        n_fft=n_fft,
                                        win_length=win_length,
                                        hop_length=hop_length)
        return reduced_noise

    def speak(self, lyrics, OutPath):
        """
        Generate speech from given lyrics and save to file.

        Args:
            lyrics (str): Text to convert to speech.
            out_path (str): Path to save the output speech file.
        """
        if (self.lang in ['ga', 'gle']) and (self.use_neon):
            wavresult = self.tts.get_audio(lyrics,  audio_format="ipython")
            wavfile.write(OutPath, rate=wavresult['rate'], data=np.array(wavresult['data']))
            self.speaker_id = 'N/A'
        elif (self.lang in ['ga', 'gle']):
            synth = self.tts.synthesizer
            sampling_rate = synth.output_sample_rate
            # random select a singer
            singer = random.choice(self.tts.speakers)
            self.speaker_id = singer
            irish_waveform = synth.tts(lyrics, speaker_name=singer)
            irish_waveform = np.array(irish_waveform)
            irish_waveform = np.squeeze(irish_waveform)
            irish_waveform = np.int16(irish_waveform * 32767)
            if self.denoise:
                irish_waveform = self.denoiser(irish_waveform,
                                               sr=sampling_rate,
                                               **self.denoise_kwargs)

            wavfile.write(OutPath, rate=sampling_rate, data=irish_waveform)
        else:
            self.speaker_id = 'N/A'
            self.tts.tts_to_file(
                text=lyrics,
                # speaker=self.tts.speakers[0],
                language=self.lang,
                speaker_wav="assets/english_bram.wav",
                file_path=OutPath
            )


    def convert(self):
        """
        Convert the voice of a speaker to another speaker.

        Args:
            source_wav (str): Path to the source wav file.
            target_wav (str): Path to the target wav file.
            file_path (str): Path to save the output wav file.
        """
        self.tts.voice_conversion_to_file(
            target_wav="assets/sweet_scarlet.wav",
            source_wav="assets/voice_conversion_target.wav",
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
