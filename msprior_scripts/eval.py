import os
import pathlib
from random import randint
from time import sleep

import gin
import numpy as np
import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from msprior.dataset import SequenceDataset
from msprior.scripted import ScriptedPrior

gin.enter_interactive_mode()

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS
DEVICE = torch.device('cpu')
RUNS = []

flags.DEFINE_integer(
    "num_steps",
    256,
    help="Number of decode steps",
)
flags.DEFINE_integer(
    'sleep',
    default=10,
    help='Number of minutes to wait between evals',
)
flags.DEFINE_integer('num_evals',
                     default=-1,
                     help='Number of eval steps to perform.')
flags.DEFINE_float('temperature',
                   default=.7,
                   help='Temperature used to perform sampling')


def add_blip(audio_signal: np.ndarray, start_pos: int, n_signal: int, sr: int):
    x = np.arange(0, n_signal)
    x = np.sin(2 * np.pi * 440 / sr * x)
    x = x * np.exp(np.linspace(0, -30, n_signal))
    audio_signal[..., start_pos:start_pos + n_signal] += x

    x = np.arange(0, n_signal)
    x = np.sin(2 * np.pi * 659.25 / sr * x)
    x = x * np.exp(np.linspace(0, -30, n_signal))
    audio_signal[...,
                 start_pos + sr // 20:start_pos + sr // 20 + n_signal] += x


def update_runs():
    global RUNS
    configs = map(str, pathlib.Path('runs').rglob('config.gin'))

    def is_prior_config(config):
        with open(config, "r") as config:
            config = config.read()
        config = config.split("\n")
        for line in config:
            if "model.Prior" in line:
                return True
        return False

    configs = filter(is_prior_config, configs)
    path = map(os.path.dirname, configs)
    path = filter(lambda x: '/eval' not in x, path)
    RUNS = list(path)


def main(argv):

    writers = {}

    step = 0
    print('Starting model evaluation')
    while step < FLAGS.num_evals or FLAGS.num_evals < 0:
        update_runs()
        for run, audio_eval, sr in eval_models():
            if run not in writers:
                writers['run'] = SummaryWriter(
                    log_dir=os.path.join(run, 'eval'))
            writer = writers['run']
            for name, audio in audio_eval.items():
                writer.add_audio(name, audio, step, sr)
            writer.flush()
        step += 1
        sleep(FLAGS.sleep * 60)


def eval_models():

    @gin.configurable
    def get_rave(path):
        return torch.jit.load(path).eval().to(DEVICE)

    @gin.configurable
    def get_dataset(path):
        return SequenceDataset(path, FLAGS.num_steps * 2)

    models = zip(RUNS, map(lambda run: ScriptedPrior(run).to(DEVICE), RUNS))

    seq = None
    for run, model in models:
        # DEFINE TEMPERATURE
        model.set_temperature(FLAGS.temperature)

        # LOAD DATASET AND RAVE
        gin.parse_config('get_rave.path = %PRETRAINED_RAVE')
        gin.parse_config('get_dataset.path = %DB_PATH')

        dataset = get_dataset()
        rave = get_rave()

        # SAMPLE FROM DATASET
        if seq is None:
            seq = dataset[randint(0, len(dataset) - 1)]

            seq_len = 0
            # FORMAT INPUT TO NN~ SPEC
            for k, v in seq.items():
                v = torch.from_numpy(v).float()
                if len(v.shape) == 2:
                    v = v.permute(1, 0)
                    seq_len = max(v.shape[-1], seq_len)
                else:
                    v = v[..., None]
                seq[k] = v.to(DEVICE)[None]

            # MAYBE PREPROCESS INPUT TOKENS
            num_steps = FLAGS.num_steps
            seq['rave'] = seq['rave'][:, :model._num_quantizers]

        # BUILD CAT INPUT
        inputs = [seq['rave']]
        if model._conditionings is not None:
            for k in model._conditionings:
                inputs.append(seq[k].expand(*seq[k].shape[:-1], seq_len))
        inputs = torch.cat(inputs, 1)

        # LISTEN TO PROMPT
        model.set_listen(True)
        pred_listen = model(inputs[..., :num_steps])[:, :-2]

        # CONTINUATION
        model.set_listen(False)
        pred_generate = model(inputs[..., num_steps:])[:, :-2]

        # SYNTHESIS
        pred = torch.cat([pred_listen, pred_generate], -1)

        audio = rave.decode(pred).cpu().numpy().reshape(-1)

        compression = rave.encode_params[-1].long().item()

        add_blip(audio, num_steps * compression, rave.sr // 2, rave.sr)
        continuation = audio.reshape(-1)

        # EXPORT AUDIO
        audio_eval = {
            'continuation': continuation,
        }

        yield run, audio_eval, rave.sr


if __name__ == "__main__":
    app.run(main)
