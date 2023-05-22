import os

import cached_conv as cc
import torch
from absl import flags

from msprior.scripted import ScriptedPrior

torch.set_grad_enabled(False)


def main(argv):
    # DUE TO THE COMPUTATIONAL COMPLEXITY OF THE MODEL, RESTRICTING TO A BATCH_SIZE OF 1
    cc.MAX_BATCH_SIZE = 1
    cc.use_cached_conv(True)

    model = ScriptedPrior(
        run=FLAGS.run,
        temporal_ratio=FLAGS.temporal_ratio,
        from_continuous=FLAGS.continuous,
        vae_path=FLAGS.vae_path,
    )
    model.apply_full_reset()
    model_name = os.path.basename(os.path.normpath(FLAGS.run)) + '.ts'
    export_path = os.path.join(FLAGS.run, model_name)
    model.export_to_ts(export_path)
    print(f'model exported to {export_path}')


FLAGS = flags.FLAGS
flags.DEFINE_string('run', default=None, required=True, help='Run to export')
flags.DEFINE_string('vae_path', default=None, help='Pretrained semantic VAE')
flags.DEFINE_integer(
    'temporal_ratio',
    default=1024,
    help='Sequence temporal ratio',
)
flags.DEFINE_bool(
    'continuous',
    default=False,
    help=
    'dequantize predictions if rave model is continuous rather than discrete',
)
