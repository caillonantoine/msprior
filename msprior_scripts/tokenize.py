import os
import pathlib
from typing import Callable, Dict, Optional, Union

import gin
import GPUtil
import lmdb
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from absl import flags
from pytorch_lightning import callbacks
from rave.quantization import VectorQuantization
from tqdm import tqdm
from udls.generated import AudioExample

from msprior.attention import TransformerDecoder
from msprior.contrastive import QuantizerBottleneck
from msprior.preprocessor import DTYPE_TO_PRECISION, PRECISION_TO_DTYPE
from msprior.utils import load_run

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the contrastive run to tokenize',
                    required=True)
flags.DEFINE_integer('batch_size',
                     default=32,
                     help='Batch size to use during tokenization')
flags.DEFINE_multi_integer('gpu', default=None, help='Gpu to use')
flags.DEFINE_integer('db_size',
                     default=100,
                     help='Maximum size of the dataset (GB)')


def main(argv):
    update()


@torch.no_grad()
def update():
    out = get_model_and_datasets()
    gpus = FLAGS.gpu or GPUtil.getAvailable(maxMemory=.05, limit=1)
    device = torch.device(f'cuda:{gpus[0]}')

    key_name = f'contrastive_{out["ratios"]}'

    tokenizer = out['tokenizer']

    db = lmdb.open(out['db_path'], map_size=1024**3 * FLAGS.db_size)

    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    def load_value(key):
        with db.begin() as txn:
            value = txn.get(key)
        return value

    values = map(load_value, keys)
    keyvalues = zip(keys, values)
    keyvalues = tqdm(keyvalues, total=len(keys))

    for kv in batch(keyvalues, batch_size=FLAGS.batch_size):
        batch_keys, batched_values = list(zip(*kv))
        examples = list(map(AudioExample.FromString, batched_values))
        dict_values = map(ae_to_dict, examples)
        list_rave = [v['rave'] for v in dict_values]
        list_rave = [torch.from_numpy(v).float() for v in list_rave]
        batch_rave = {'rave': torch.stack(list_rave, 0).to(device)}

        tokens = tokenizer(batch_rave).cpu().numpy().astype(np.int16)

        with db.begin(write=True) as txn:
            for key, audio_example, token in zip(batch_keys, examples, tokens):
                buffer = audio_example.buffers[key_name]
                buffer.data = token.tobytes()
                buffer.shape[:] = token.shape
                buffer.precision = DTYPE_TO_PRECISION[np.int16]
                txn.put(key, audio_example.SerializeToString())

    db.close()


def ae_to_dict(ae: Union[AudioExample, bytes]):
    if isinstance(ae, bytes):
        ae = AudioExample.FromString(ae)

    example = {}
    for k in ae.buffers:
        buffer = ae.buffers[k]
        array = np.frombuffer(
            buffer.data,
            dtype=PRECISION_TO_DTYPE[buffer.precision]).reshape(buffer.shape)
        example[k] = array.astype(np.float32)
    return example


def batch(iterable, batch_size):
    batch = []
    for elm in iterable:
        batch.append(elm)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_model_and_datasets(run: Optional[str] = None,
                           batch_size: Optional[int] = None):
    run, model = load_run(run or FLAGS.run)

    @gin.configurable
    def get_dataset(dataset):
        return dataset

    @gin.configurable
    def get_ratios(ratios):
        return int(np.prod(ratios))

    @gin.configurable
    def get_db_path(db_path):
        return db_path

    gin.parse_config('get_dataset.dataset = @SequenceDataset()')
    gin.parse_config('get_ratios.ratios = %RATIOS')
    gin.parse_config('get_db_path.db_path = %DB_PATH')

    dataset = get_dataset()
    split = min(1000, len(dataset) // 100)
    train, val = torch.utils.data.random_split(
        dataset,
        (len(dataset) - split, split),
    )
    train = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size or FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )
    val = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size or FLAGS.batch_size,
        num_workers=16,
    )

    @torch.no_grad()
    def tokenizer(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        decoder = model.temporal_model_mlm.decoder
        decoder: TransformerDecoder

        for layer_index, layer in enumerate(decoder.layers):
            if isinstance(layer, QuantizerBottleneck):
                quantizer = layer
                quantizer: QuantizerBottleneck
                break

        device = batch['rave'].device
        model.eval()
        if model.device != device:
            model.to(device)

        features = model.encode(batch)
        hidden = model.temporal_model_contrastive(features)
        hidden = model.temporal_model_mlm(hidden, early_stop=layer_index)
        indices = quantizer.tokenize(hidden)

        return indices

    return {
        'run': run,
        'tokenizer': tokenizer,
        'train': train,
        'val': val,
        'dataset': dataset,
        'ratios': get_ratios(),
        'db_path': get_db_path(),
    }
