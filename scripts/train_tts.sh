MODEL_FOLDER=/media/bramiozo/DATA-FAST/TTS/tts_models/gle/tts-vits-cv-ga_seanos_6000
TTS_FOLDER=/media/bramiozo/Storage1/bramiozo/VIRTUALENVS/Python/seanos-bFLQpzeS-py3.10/lib/python3.10/site-packages/TTS/bin
CUDA_VISIBLE_DEVICES="0" python $TTS_FOLDER/train_tts.py --config_path $MODEL_FOLDER/config.json
