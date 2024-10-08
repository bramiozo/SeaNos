import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import hifigan_config
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
import argparse
import json

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")


config = hifigan_config.HifiganConfig(
    batch_size=7,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=False,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=31744,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    lr_gen=1e-4,
    lr_disc=1.2e-4,
    data_path=os.path.join(output_path, "clips"),
    output_path=output_path,
)



if __name__=="__main__":
    # init audio processor
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default=config.data_path)
    argparser.add_argument("--output_path", type=str, default=config.output_path)
    argparser.add_argument("--eval_split_size", type=int, default=config.eval_split_size)
    argparser.add_argument("--config_path", type=str, default=None)
    parse = argparser.parse_args()

    if parse.config_path is not None:
        # parse config path, extract json in dictionary and load
        print("Loading dictionary for config")
        f = open(parse.config_path, 'r')
        config_dict = json.load(f)
        config = hifigan_config.HifiganConfig(**config_dict)
        
    config.eval_split_size = parse.eval_split_size
    config.data_path = parse.data_path
    output_path = parse.output_path

    config_audio = {
         "fft_size": 2048,
        "sample_rate": 44100,
        "win_length": 2048,
        "hop_length": 512,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 22000,
        "frame_shift_ms": None,
        "frame_length_ms": None,
        "stft_pad_mode": "reflect",
        "resample": False,
        "preemphasis": 0.0,
        "ref_level_db": 20,
        "do_sound_norm": False,
        "log_func": "np.log10",
        "do_trim_silence": True,
        "trim_db": 45,
        "do_rms_norm": False,
        "db_level": None,
        "power": 1.5,
        "griffin_lim_iters": 60,
        "spec_gain": 20,
        "do_amp_to_db_linear": True,
        "do_amp_to_db_mel": True,
        "pitch_fmax": 640.0,
        "pitch_fmin": 1.0,
        "signal_norm": True,
        "min_level_db": -100,
        "symmetric_norm": True,
        "max_norm": 4.0,
        "clip_norm": True,
        "stats_path": None
    }
    print(type(config))
    print(config)

    config.l1_spec_loss_params =  {
        "use_mel": True,
        "sample_rate": 44100,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
        "n_mels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 22000
    }

    config.audio = {
        "fft_size": 2048,
        "sample_rate": 44100,
        "win_length": 2048,
        "hop_length": 512,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 22000
    }
    
    ap = AudioProcessor(**config.audio)

    # load training samples
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

    # init model
    model = GAN(config, ap)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()
