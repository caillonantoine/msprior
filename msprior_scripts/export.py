import os

import cached_conv as cc
import torch
from absl import flags

from msprior.scripted import ScriptedPrior

torch.set_grad_enabled(False)


def main(argv):
    cc.MAX_BATCH_SIZE = FLAGS.batch_size
    cc.use_cached_conv(True)

    model = ScriptedPrior(
        run=FLAGS.run,
        temporal_ratio=FLAGS.temporal_ratio,
        from_continuous=FLAGS.continuous,
        vae_path=FLAGS.vae_path,
        ema_weights=FLAGS.ema_weights,
    )
    model.apply_full_reset()
    model_name = os.path.basename(os.path.normpath(FLAGS.run)) + '.ts'
    export_path = os.path.join(FLAGS.run, model_name)
    model.export_to_ts(export_path)
    print(f'model exported to {export_path}')


FLAGS = flags.FLAGS
flags.DEFINE_string('run', default=None, required=True, help='Run to export')
flags.DEFINE_string('vae_path', default=None, help='Pretrained semantic VAE')
flags.DEFINE_integer('batch_size', default=1, help='Maximum batch size')
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if available')

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
