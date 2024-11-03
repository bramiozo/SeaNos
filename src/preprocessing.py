# Script cleaning training data
# equalisation
# noise removal

import argparse
import os
from multiprocessing.managers import Value

import numpy as np
from scipy.signal import iirnotch, lfilter, butter, savgol_filter, fftconvolve
from pydub import AudioSegment
import librosa
from scipy.ndimage import convolve1d

from typing import List, Union, Tuple, Literal

# Add RIR?

def add_gaussian_noise(audio_segment, noise_level=0.005):
    """
    Add Gaussian noise to an audio segment.

    :param audio_segment: PyDub AudioSegment
    :param noise_level: relative magnitude of the gaussian noise
    :return:
    """
    # Convert audio segment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level * np.max(np.abs(samples)), samples.shape)

    # Add noise to the samples
    noisy_samples = samples + noise

    # Ensure the values are within the valid range for the audio format
    noisy_samples = np.clip(noisy_samples, -32768, 32767).astype(np.int16)

    # Create a new AudioSegment from the noisy samples
    noisy_audio = AudioSegment(
        noisy_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    return noisy_audio


def stereo_to_mono(audio_segment):
    """
    Convert a stereo audio segment to mono by averaging the channels.

    :param audio_segment: PyDub AudioSegment
    :return: Mono PyDub AudioSegment
    """
    return audio_segment.set_channels(1)


def resampler(audio_segment, sampling_rate=44050):
    """
    Resample the audio to the specified sampling rate.

    :param audio_segment: PyDub AudioSegment
    :param sampling_rate: Target sampling rate (default: 44000 Hz)
    :return: Resampled PyDub AudioSegment
    """
    # Convert PyDub AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples(), dtype=float)

    # Resample using librosa
    resampled = librosa.resample(samples, orig_sr= audio_segment.frame_rate, target_sr= sampling_rate)

    # Convert back to PyDub AudioSegment
    resampled_segment = AudioSegment(
        resampled.astype(np.int16).tobytes(),
        frame_rate=sampling_rate,
        sample_width=2,
        channels=audio_segment.channels
    )

    return resampled_segment
def noise_reduction(audio_segment: AudioSegment,
                    noise_freq: int=64,
                    q: float=30.0):
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


def kernel_smoother(audio_segment: AudioSegment, smoothing_kernel=(1, 2, 1)):
    '''
    Applies Gaussian smoother to audio signal

    :param audio_segment: The input audio segment to be smoothed
    :param kernel: The weights of the Gaussian filter
    :return: Smoothed AudioSegment
    '''
    # Convert AudioSegment to raw audio data (numpy array)
    samples = np.array(audio_segment.get_array_of_samples())

    # Apply Gaussian smoothing kernel using convolution
    smoothed_samples = convolve1d(samples, weights=smoothing_kernel, mode='reflect')

    # Ensure smoothed samples are within the original data type range
    smoothed_samples = np.clip(smoothed_samples, -32768, 32767).astype(np.int16)

    # Convert the smoothed numpy array back to an AudioSegment
    smoothed_audio_segment = AudioSegment(
        smoothed_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )
    return smoothed_audio_segment


def special_smoother(audio_segment: AudioSegment,
                     kernel_type: Literal['savgol_filter', 'fftconvolve'] = 'savgol_filter',
                     window_length:int=9,
                     polyorder: int=3
                     ):
    '''
    Applies a special smoother to the audio signal using either Savitzky-Golay filter or FFT convolution

    :param audio_segment: The input audio segment to be smoothed
    :param smoothing_kernel: The type of smoothing to apply, either 'savgol_filter' or 'fftconvolve'
    :return: Smoothed AudioSegment
    '''
    # Convert AudioSegment to raw audio data (numpy array)
    samples = np.array(audio_segment.get_array_of_samples())

    # Choose smoothing method
    if kernel_type == 'savgol_filter':
        # Apply Savitzky-Golay filter (e.g., window_length=window_length, polyorder=2)
        smoothed_samples = savgol_filter(samples, window_length=window_length, polyorder=polyorder)

    elif kernel_type == 'fftconvolve':
        # Define a Gaussian-like kernel for FFT convolution
        kernel = np.exp(-np.linspace(-3, 3, window_length) ** 2)
        kernel /= kernel.sum()  # Normalize the kernel
        # Apply FFT convolution with the kernel
        smoothed_samples = fftconvolve(samples, kernel, mode='same')

    else:
        raise ValueError("smoothing_kernel must be either 'savgol_filter' or 'fftconvolve'")

    # Ensure the smoothed samples are within the range for 16-bit audio
    smoothed_samples = np.clip(smoothed_samples, -32768, 32767).astype(np.int16)

    # Convert the smoothed samples back to an AudioSegment
    smoothed_audio_segment = AudioSegment(
        smoothed_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    return smoothed_audio_segment

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


def low_pass_filter(audio_segment, cutoff_freq=4_000):
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
    :return: Trimmed PyDub AudioSegment+
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
                  use_noise_reduction,
                  use_high_pass,
                  use_low_pass,
                  use_autotrim,
                  use_pitcher,
                  use_mono,
                  use_resampler,
                  use_noise_adder,
                  use_kernel_smoother,
                  use_special_smoother,
                  noise_freq=60, octave_change=0.25,
                  target_sample_rate=44_000, noise_level=1e-4,
                  smoothing_kernel=(1,2,1),
                  poly_order=3,
                  kernel_type: Literal['savgol_filter', 'fftconvolve']='savgol_filter',
                  window_length=7
                  ):
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

    if isinstance(processed_audio, np.ndarray):
        if len(processed_audio.shape) == 1:
            num_channels = 1  # Mono
        else:
            num_channels = processed_audio.shape[1]
    elif isinstance(processed_audio, AudioSegment):
        num_channels = processed_audio.channels
    else:
        raise ValueError("Processed audio should be an AudioSegment instance, or a numpy array")

    if use_mono and num_channels > 1:
        processed_audio = stereo_to_mono(processed_audio)
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
    if use_resampler:
        processed_audio = resampler(processed_audio, target_sample_rate)
    if use_noise_adder:
        processed_audio = add_gaussian_noise(processed_audio, noise_level=noise_level)
    if use_kernel_smoother:
        processed_audio = kernel_smoother(processed_audio, smoothing_kernel=smoothing_kernel)
    if use_special_smoother:
        processed_audio = special_smoother(processed_audio, kernel_type=kernel_type,
                                           window_length=window_length, polyorder=poly_order)

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
    parser.add_argument("--add-noise", dest="add_noise", action="store_true", default=False,
                        help="Add Gaussian noise to the waveform")
    parser.add_argument("--noise-level", dest="noise_level", type=float, default=1e-5,
                        help="Magnitude of the any Gaussian noise added")


    parser.set_defaults(use_noise_reduction=True, use_high_pass=True, use_low_pass=True, use_autotrim=True)

    args = parser.parse_args()
    kwargs = {'noise_freq': float(args.noise_freq),
              'octave_change': float(args.octave_change),
              'target_sample_rate': args.target_sample_rate,
              'noise_level': float(args.noise_level)
              }

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
                                            args.add_noise,
                                            **kwargs
                                            )
            processed_audio.export(output_path, format="wav")
            print(f"Saved processed audio to {output_path}")

    print("All files processed successfully.")


if __name__ == "__main__":
    main()