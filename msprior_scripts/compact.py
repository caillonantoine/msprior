import os
import pathlib
import shutil
from os import path

import lmdb
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db',
    default=None,
    required=True,
    help='database to compact',
)
flags.DEFINE_integer('max_elm', default=None, help='max number of elements.')


def main(argv):
    db = lmdb.open(FLAGS.db, readonly=True)

    target = path.normpath(FLAGS.db) + "_compact"
    os.makedirs(target, exist_ok=False)

    pretrained = next(iter(pathlib.Path(FLAGS.db).rglob("*.ts")))
    pretrained = str(pretrained)
    pretrained_name = path.basename(pretrained)
    shutil.copy(pretrained, path.join(target, pretrained_name))

    target = lmdb.open(
        target,
        map_size=100 * 1024**3,
    )

    num_elm = 0
    with db.begin() as source_txn:
        keys = list(source_txn.cursor().iternext(values=False))
        with target.begin(write=True) as target_txn:
            for k, v in tqdm(
                    source_txn.cursor().iternext(),
                    desc="Compacting",
                    total=len(keys),
            ):
                target_txn.put(k, v)
                num_elm += 1
                if FLAGS.max_elm is not None:
                    if num_elm >= FLAGS.max_elm:
                        break


if __name__ == "__main__":
    app.run(main)
