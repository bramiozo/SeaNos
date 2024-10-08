DATA_FOLDER=/media/bramiozo/DATA-FAST/TTS/tts_models/gle/seannos_datasource
SCRIPT_FOLDER=/media/bramiozo/Storage1/bramiozo/DEV/GIT/B-lab/SeaNos
CONFIG_FILE=/media/bramiozo/DATA-FAST/TTS/tts_models/gle/hifigan_vocoder_seanos/config.json
CUDA_VISIBLE_DEVICES="0" python $SCRIPT_FOLDER/src/train_vocoder.py --data_path $DATA_FOLDER/clips --output_path output
