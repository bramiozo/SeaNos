DATA_FOLDER=/media/bramiozo/DATA-FAST/TTS/tts_models/gle/tts-vits-cv-ga_seanos_1200
SCRIPT_FOLDER=/media/bramiozo/Storage1/bramiozo/DEV/GIT/B-lab/SeaNos
CUDA_VISIBLE_DEVICES="0" python $SCRIPT_FOLDER/src/train_vocoder.py --data_path $DATA_FOLDER/datasets/clips --output_path output
