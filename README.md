# SeaNos

Development of ```Text``` $\rightarrow$ ```Sean Nos``` using existing models.


# Technologies text-to-speech

## **TTS with MMS**

[TTS](https://huggingface.co/coqui/XTTS-v2), [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)


Download the Irish model
```wget https://dl.fbaipublicfiles.com/mms/tts/gle.tar.gz # Irish gle)```

```pip install TTS ```
```python

from TTS.api import TTS
tts = TTS("loc/to/irish/model", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")
```
