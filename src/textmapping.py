
# https://pypi.org/project/phonemizer/
# https://github.com/facebookresearch/fairseq/blob/main/examples/mms/tts/tutorial/MMS_TTS_Inference_Colab.ipynb

class Phonetics():
    def __init__(self, text, source_lang, target_lang):
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = Translator()

    def translate(self):
        return self.translator.translate(self.text, src=self.source_lang, dest=self.target_lang).text

    def generate(self):
        return self.translate()
