from typing import Callable, Dict, Optional, Tuple

import gin
import lmdb
import torch
from torch.utils.data import Dataset, Subset
from udls import AudioExample

TensorDict = Dict[str, torch.Tensor]


def ordered_dataset_split(dataset: Dataset,
                          sizes: Tuple[int, ...]) -> Tuple[Dataset]:
    assert sum(sizes) == len(dataset)
    datasets = []
    start_idx = 0
    for size in sizes:
        datasets.append(Subset(dataset, range(start_idx, start_idx + size)))
        start_idx += size
    return tuple(datasets)


@gin.configurable
class SequenceDataset(Dataset):

    def __init__(
            self,
            db_path: str,
            task_fn: Optional[Callable[[TensorDict],
                                       TensorDict]] = None) -> None:
        super().__init__()
        self._env = None
        self._keys = None
        self._db_path = db_path
        self.task_fn = task_fn

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
        ae = ae.as_dict()
        if self.task_fn is not None:
            ae = self.task_fn(ae)
        return ae

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(
                self._db_path,
                lock=False,
                readahead=False,
            )
        return self._env

    @property
    def keys(self):
        if self._keys is None:
            with self.env.begin(write=False) as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys
