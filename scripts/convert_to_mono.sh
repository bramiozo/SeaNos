for file in "/media/bramiozo/DATA-FAST/TTS/tts_models/gle/tts-vits-cv-ga_seanos/datasets/clips"/*.wav; do
	ffmpeg -i "$file" -ac 1 "${file%.wav}_mono.wav"
done


