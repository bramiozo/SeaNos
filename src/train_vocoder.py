import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import hifigan_config
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
import argparse

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")

config = hifigan_config.HifiganConfig(
    batch_size=22,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=False,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(output_path, "clips"),
    output_path=output_path,
)



if __name__=="__main__":
    # init audio processor
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default=config.data_path)
    argparser.add_argument("--output_path", type=str, default=config.output_path)
    argparser.add_argument("--eval_split_size", type=int, default=config.eval_split_size)
    parse = argparser.parse_args()

    config.eval_split_size = parse.eval_split_size
    config.data_path = parse.data_path
    output_path = parse.output_path

    ap = AudioProcessor(**config.audio.to_dict())

    # load training samples
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

    # init model
    model = GAN(config, ap)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()