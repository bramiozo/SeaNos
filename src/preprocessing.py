# Script cleaning training data
# equalisation
# noise removal

import argparse
import os
import numpy as np
from scipy.signal import iirnotch, lfilter, butter
from pydub import AudioSegment
import librosa

def stereo_to_mono(audio_segment):
    """
    Convert a stereo audio segment to mono by averaging the channels.

    :param audio_segment: PyDub AudioSegment
    :return: Mono PyDub AudioSegment
    """
    return audio_segment.set_channels(1)


def resampler(audio_segment, sampling_rate=44000):
    """
    Resample the audio to the specified sampling rate.

    :param audio_segment: PyDub AudioSegment
    :param sampling_rate: Target sampling rate (default: 44000 Hz)
    :return: Resampled PyDub AudioSegment
    """
    # Convert PyDub AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    # Resample using librosa
    resampled = librosa.resample(samples, audio_segment.frame_rate, sampling_rate)

    # Convert back to PyDub AudioSegment
    resampled_segment = AudioSegment(
        resampled.astype(np.int16).tobytes(),
        frame_rate=sampling_rate,
        sample_width=2,
        channels=1 if audio_segment.channels == 1 else 2
    )

    return resampled_segment
def noise_reduction(audio_segment, noise_freq, q=30.0):
    """
    Reduce noise at a specific frequency using a notch filter.

    :param audio_segment: PyDub AudioSegment
    :param noise_freq: Frequency to filter out
    :param q: Quality factor. Higher q gives a narrower notch
    :return: Filtered PyDub AudioSegment
    """
    sample_rate = audio_segment.frame_rate
    samples = np.array(audio_segment.get_array_of_samples())

    # Create a notch filter
    nyq = 0.5 * sample_rate
    freq = noise_freq / nyq
    b, a = iirnotch(freq, q)

    filtered_samples = lfilter(b, a, samples)

    filtered_audio = AudioSegment(
        filtered_samples.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=audio_segment.channels
    )

    return filtered_audio


def high_pass_filter(audio_segment, cutoff_freq=200):
    """
    Apply a high pass filter to the audio. This filters out all frequencies below the cutt-off frequency

    :param audio_segment: PyDub AudioSegment
    :param cutoff_freq: Cutoff frequency for the filter (default: 500 Hz)
    :return: Filtered PyDub AudioSegment
    """
    sample_rate = audio_segment.frame_rate
    samples = np.array(audio_segment.get_array_of_samples())

    # Ensure cutoff frequency is less than Nyquist frequency
    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.99

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist

    b, a = butter(5, normalized_cutoff, btype='high')
    filtered_samples = lfilter(b, a, samples)

    filtered_audio = AudioSegment(
        filtered_samples.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=audio_segment.channels
    )

    return filtered_audio


def low_pass_filter(audio_segment, cutoff_freq=6_000):
    """
    Apply a low pass filter to the audio. This filters out all frequencies above the cutt-off frequency

    :param audio_segment: PyDub AudioSegment
    :param cutoff_freq: Cutoff frequency for the filter (default: 4000 Hz)
    :return: Filtered PyDub AudioSegment
    """
    sample_rate = audio_segment.frame_rate
    samples = np.array(audio_segment.get_array_of_samples())

    # Ensure cutoff frequency is less than Nyquist frequency
    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.99

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist

    b, a = butter(5, normalized_cutoff, btype='low')
    filtered_samples = lfilter(b, a, samples)

    filtered_audio = AudioSegment(
        filtered_samples.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=audio_segment.channels
    )

    return filtered_audio


def change_pitch(audio_segment, octave_change=0.5):
    frameRate = audio_segment.frame_rate
    new_sample_rate = int(frameRate * (2 ** octave_change))
    new_sound = audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_sample_rate})
    return new_sound.set_frame_rate(new_sample_rate)


def autotrim(audio_segment, silence_threshold=-40, chunk_size=10):
    """
    Trim leading and trailing silence of audioclips

    :param audio_segment: PyDub AudioSegment
    :param silence_threshold: The threshold (in dB) below which to consider as silence
    :param chunk_size: Size of audio chunks to analyze (in milliseconds)
    :return: Trimmed PyDub AudioSegment
    """
    def detect_leading_silence(segment, silence_threshold, chunk_size):
        trim_ms = 0
        while segment[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(segment):
            trim_ms += chunk_size
        return trim_ms

    start_trim = detect_leading_silence(audio_segment, silence_threshold, chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold, chunk_size)

    duration = len(audio_segment)
    trimmed_audio = audio_segment[start_trim:duration-end_trim]

    return trimmed_audio


def process_audio(audio_segment,
                  use_noise_reduction, use_high_pass, use_low_pass,
                  use_autotrim, use_pitcher, use_mono, use_resampler,
                  noise_freq=60, octave_change=0.25, target_sample_rate=44_000):
    """
    Apply selected processing steps to the audio segment.

    :param audio_segment: PyDub AudioSegment
    :param use_noise_reduction: Boolean flag for noise reduction
    :param use_high_pass: Boolean flag for high-pass filter
    :param use_low_pass: Boolean flag for low-pass filter
    :param use_autotrim: Boolean flag for auto-trimming
    :param use_pitcher: Boolean flag for pitch change
    :param use_mono: Boolean flag for stereo to mono conversion
    :param use_resampler: Boolean flag for resampling
    :param noise_freq: Frequency to filter out in noise reduction
    :param octave_change: Octave change for pitch shifting
    :param target_sample_rate: Target sample rate for resampling
    :return: Processed PyDub AudioSegment
    """
    processed_audio = audio_segment

    if use_noise_reduction:
        processed_audio = noise_reduction(processed_audio, noise_freq)  # Example: reduce 60Hz hum
    if use_high_pass:
        processed_audio = high_pass_filter(processed_audio)
    if use_low_pass:
        processed_audio = low_pass_filter(processed_audio)
    if use_autotrim:
        processed_audio = autotrim(processed_audio)
    if use_pitcher:
        processed_audio = change_pitch(processed_audio, octave_change)
    if use_mono and processed_audio.channels > 1:
        processed_audio = stereo_to_mono(processed_audio)
    if use_resampler:
        processed_audio = resampler(processed_audio, target_sample_rate)

    return processed_audio


def main():
    parser = argparse.ArgumentParser(description="Process audio files in a directory.")
    parser.add_argument("--input-dir", help="Input directory containing .wav files",
                        dest="input_dir")
    parser.add_argument("--output-dir", help="Output directory for processed .wav files",
                        dest="output_dir")
    parser.add_argument("--filename-prefix", help="", default="processed", dest="prefix")
    parser.add_argument("--no-noise-reduction", action="store_false", dest="use_noise_reduction",
                        help="Disable noise reduction")
    parser.add_argument("--no-high-pass", action="store_false", dest="use_high_pass",
                        help="Disable high-pass filter")
    parser.add_argument("--no-low-pass", action="store_false", dest="use_low_pass",
                        help="Disable low-pass filter")
    parser.add_argument("--no-autotrim", action="store_false", dest="use_autotrim",
                        help="Disable auto-trimming")
    parser.add_argument("--pitcher", dest="use_pitcher",action="store_true", default=False,
                        help="Enable pitch change, requires setting --octave_change, otherwise there is no change")
    parser.add_argument("--octave-change", dest="octave_change", default=0.)
    parser.add_argument("--noise_freq", dest="noise_freq", default=60)
    parser.add_argument("--mono", dest="use_mono", action="store_true", default=False,
                        help="Convert stereo to mono")
    parser.add_argument("--resample", dest="use_resampler", action="store_true", default=False,
                        help="Resample audio")
    parser.add_argument("--target-sample-rate", dest="target_sample_rate", type=int, default=44_000,
                        help="Target sample rate for resampling")

    parser.set_defaults(use_noise_reduction=True, use_high_pass=True, use_low_pass=True, use_autotrim=True)

    args = parser.parse_args()
    kwargs = {'noise_freq': float(args.noise_freq),
              'octave_change': float(args.octave_change),
              'target_sample_rate': args.target_sample_rate}

    assert(args.input_dir!=args.output_dir), "Input directory cannot be the same as the output directory!"
    assert([fc in args.prefix for fc in [',', '_']]), "Prefix cannot contain ',' or '_'"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all .wav files in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"{args.prefix}_{filename}")

            print(f"Processing {filename}...")
            audio = AudioSegment.from_wav(input_path)
            processed_audio = process_audio(audio,
                                            args.use_noise_reduction,
                                            args.use_high_pass,
                                            args.use_low_pass,
                                            args.use_autotrim,
                                            args.use_pitcher,
                                            args.use_mono,
                                            args.use_resampler,
                                            **kwargs
                                            )
            processed_audio.export(output_path, format="wav")
            print(f"Saved processed audio to {output_path}")

    print("All files processed successfully.")


if __name__ == "__main__":
    main()