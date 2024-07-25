#!/bin/bash

# Directory containing the audio files
audio_dir="/media/bramiozo/DATA-FAST/TTS/tts_models/gle/tts-vits-cv-ga_seanos/datasets/clips"

# Loop over all files with _mono.wav suffix in the directory
for file in "$audio_dir"/*_mono.wav; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Construct the new filename by removing the _mono suffix
        new_file="${file/_mono.wav/.wav}"
        # Rename the file
        mv "$file" "$new_file"
        echo "Renamed '$file' to '$new_file'"
    else
        echo "No files with _mono suffix found in the specified directory."
    fi
done

