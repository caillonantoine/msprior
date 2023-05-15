import dataclasses
import functools
import importlib
import itertools
import math
import multiprocessing
import os
import pathlib
import shutil
from datetime import timedelta
from functools import partial
from typing import Sequence, Tuple

import GPUtil
import lmdb
import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
from rave.dataset import normalize_signal
from tqdm import tqdm

import msprior.preprocessor
import msprior.utils

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string('rave',
                    default=None,
                    required=True,
                    help='Path to the pretrained rave model')
flags.DEFINE_string('out_path',
                    default=None,
                    required=True,
                    help='Output dataset path')
flags.DEFINE_string('audio',
                    default=None,
                    required=True,
                    help='Path to the audio files')
flags.DEFINE_integer('num_secs',
                     default=16,
                     help='Audio chunk size in seconds')
flags.DEFINE_integer('num_tokens', default=1024, help='Rave vocabulary size')
flags.DEFINE_integer('batch_size',
                     default=64,
                     help='Batch size during processing')
flags.DEFINE_multi_string('preprocessor',
                          default=[],
                          help='Additional preprocessors to import')
flags.DEFINE_integer('gpu', default=None, help='GPU to use.')
flags.DEFINE_integer('db_size',
                     default=100,
                     help='Maximum size of the dataset (GB)')
flags.DEFINE_bool('dyndb',
                  default=True,
                  help='Allow the database to grow dynamically')
flags.DEFINE_bool('normalize',
                  default=False,
                  help="Normalize audio before preprocessing.")
flags.DEFINE_integer('silence_threshold',
                     default=None,
                     help="Optional silence threshold (30 - 50).")
flags.DEFINE_integer(
    'resolution',
    default=64,
    help="Discretization resolution to use when using continuous rave models.")


def process_batch(
    batched_audio_sample: Tuple[int, Sequence[Tuple[np.ndarray, str]]],
    target_size: int,
    database: lmdb.Environment,
    batch_size: int,
) -> int:
    batch_id, audio_sample = batched_audio_sample
    audio_samples, paths = zip(*audio_sample)

    if FLAGS.normalize:
        audio_samples = [normalize_signal(a) for a in audio_samples]

    audio_examples = msprior.preprocessor.preprocess_batch(
        audio_samples, paths, target_size)

    for ae_id, (ae, path) in enumerate(zip(audio_examples, paths)):
        with database.begin(write=True) as txn:
            key = f'{batch_id*batch_size + ae_id:016d}'.encode()
            txn.put(key, ae.SerializeToString())

    return batch_id


@dataclasses.dataclass
class RAVEEncoder(msprior.preprocessor.Preprocessor):
    pretrained_path: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.pretrained_path is not None
        self.model = torch.jit.load(self.pretrained_path)
        self.model.to(self.device).eval()

        self.temporal_ratio = self.model.encode_params[-1].item()
        self.from_continuous = "Discrete" not in self.model.encoder.original_name
        self.resolution = FLAGS.resolution

    def quantize(self, x):
        x = x / 2
        x = .5 * (1 + torch.erf(x / math.sqrt(2)))
        x = torch.floor(x * (self.resolution - 1))
        return x

    def __call__(self, audio_arrays, audio_paths):
        audio_arrays = np.stack(audio_arrays, 0)
        audio_batch = torch.from_numpy(audio_arrays).float().reshape(
            audio_arrays.shape[0], 1, -1).to(self.device)
        z = self.model.encode(audio_batch).permute(0, 2, 1)
        if self.from_continuous:
            z = self.quantize(z)
        z = z.cpu().numpy()
        return z


def main(argv):
    if FLAGS.gpu is not None:
        device = torch.device(f'cuda:{FLAGS.gpu}')
    elif available := GPUtil.getAvailable(maxMemory=.05, limit=1):
        device = torch.device(f'cuda:{available[0]}')
    else:
        device = torch.device('cpu')

    msprior.preprocessor.set_default_gpu(device)

    # create rave preprocessor
    FLAGS.rave = os.path.abspath(FLAGS.rave)
    rave_preprocessor = RAVEEncoder(
        'rave',
        mode='dynamic',
        dtype=np.int16,
        pretrained_path=FLAGS.rave,
        resample=False,
    )
    sampling_rate = rave_preprocessor.model.sr
    temporal_ratio = rave_preprocessor.temporal_ratio

    msprior.preprocessor.register_preprocessor(rave_preprocessor)
    msprior.preprocessor.set_sampling_rate(sampling_rate)

    # import additional preprocessors
    for preprocessor in FLAGS.preprocessor:
        importlib.import_module(preprocessor)

    # create data reader and writer
    n_signal = sampling_rate * FLAGS.num_secs
    n_signal = 2**int(math.ceil(math.log2(n_signal)))
    print(f'Using chunks of {n_signal/sampling_rate:.2f}s')

    chunk_load = partial(
        msprior.utils.load_audio_chunk,
        n_signal=n_signal,
        sr=sampling_rate,
        silenceremove=FLAGS.silence_threshold,
    )

    env = lmdb.open(
        FLAGS.out_path,
        map_size=FLAGS.db_size * 1024**3,
        readahead=False,
        writemap=not FLAGS.dyndb,
        map_async=not FLAGS.dyndb,
    )

    shutil.copy(
        FLAGS.rave,
        os.path.join(
            FLAGS.out_path,
            os.path.basename(FLAGS.rave),
        ),
    )

    pool = multiprocessing.Pool()

    # search for audio files
    wavs = itertools.chain(*[
        pathlib.Path(FLAGS.audio).rglob(f"*.{ext}")
        for ext in ['opus', 'ogg', 'wav', 'mp3', 'aiff', 'flac', 'm4a']
    ])
    wavs = list(map(str, wavs))
    print(f'Found {len(wavs)} audio files in {FLAGS.audio}')

    # load and batch chunks
    chunks = msprior.utils.flatmap(
        pool,
        chunk_load,
        wavs,
        queue_size=FLAGS.batch_size,
    )
    chunks_batch = msprior.utils.batch(chunks, batch_size=FLAGS.batch_size)
    chunks_batch = enumerate(chunks_batch)

    # process dataset
    processed_samples = map(
        functools.partial(
            process_batch,
            target_size=n_signal // temporal_ratio,
            database=env,
            batch_size=FLAGS.batch_size,
        ),
        chunks_batch,
    )

    with tqdm() as pbar:
        for batch_id in processed_samples:
            batch_id += 1
            n_seconds = n_signal / sampling_rate * batch_id * FLAGS.batch_size
            pbar.set_description(
                f'dataset length: {timedelta(seconds=n_seconds)}')

    pool.close()
    env.sync()


if __name__ == '__main__':
    app.run(main)
