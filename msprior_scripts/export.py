import os

import cached_conv as cc
import torch
from absl import flags

from msprior.scripted import ScriptedPrior

torch.set_grad_enabled(False)


def main(argv):
    cc.use_cached_conv(True)

    model = ScriptedPrior(FLAGS.run, FLAGS.temporal_ratio)
    model_name = os.path.basename(os.path.normpath(FLAGS.run)) + '.ts'
    export_path = os.path.join(FLAGS.run, model_name)
    model.export_to_ts(export_path)
    print(f'model exported to {export_path}')


FLAGS = flags.FLAGS
flags.DEFINE_string('run', default=None, required=True, help='Run to export')
flags.DEFINE_integer(
    'temporal_ratio',
    default=1024,
    help='Sequence temporal ratio',
)
