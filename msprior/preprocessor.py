import dataclasses
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pathos.multiprocessing as mp
import torch
from udls.generated import AudioExample

_PREPROCESSOR_LIST = []
_PREPROCESSOR_DEFAULT_GPU = torch.device('cpu')
_SAMPLING_RATE = 44100

DTYPE_TO_PRECISION = {
    np.int16: AudioExample.Precision.INT16,
    np.int32: AudioExample.Precision.INT32,
    np.int64: AudioExample.Precision.INT64,
    np.float16: AudioExample.Precision.FLOAT16,
    np.float32: AudioExample.Precision.FLOAT32,
    np.float64: AudioExample.Precision.FLOAT64,
}

PRECISION_TO_DTYPE = {
    AudioExample.Precision.INT16: np.int16,
    AudioExample.Precision.INT32: np.int32,
    AudioExample.Precision.INT64: np.int64,
    AudioExample.Precision.FLOAT16: np.float16,
    AudioExample.Precision.FLOAT32: np.float32,
    AudioExample.Precision.FLOAT64: np.float64,
}

Scalar = Union[int, float]
Array = np.ndarray


def interpolate(input_array: np.ndarray, size: int, mode: str) -> np.ndarray:
    input_array = torch.from_numpy(input_array).float()
    assert len(input_array.shape) < 3

    if len(input_array.shape) == 2:
        input_array = input_array.permute(0, 1)
    else:
        input_array = input_array[None, :]

    input_array = input_array[None]

    output_array = torch.nn.functional.interpolate(
        input_array,
        size=size,
        mode=mode,
    )

    output_array = output_array[0].permute(1, 0)

    return output_array.numpy()


@dataclasses.dataclass
class Preprocessor:
    name: str
    mode: Literal['static', 'dynamic']
    dtype: np.dtype
    resample: bool = True
    resampling_mode: Literal['nearest', 'linear'] = 'nearest'

    parallel: bool = False
    device: Optional[torch.device] = None
    sampling_rate: Optional[int] = None

    def __post_init__(self):
        self.device = self.device or get_default_gpu()
        self.sampling_rate = self.sampling_rate or get_sampling_rate()

    def process_single_example(self, audio_array: Array,
                               audio_path: str) -> Union[Scalar, Array]:
        raise NotImplementedError

    def __call__(self, audio_arrays: Sequence[Array],
                 audio_paths: Sequence[str]) -> Sequence[Union[Scalar, Array]]:
        if self.parallel:

            def process(args):
                return self.process_single_example(*args)

            with mp.Pool() as pool:
                outputs = list(
                    pool.map(
                        process,
                        zip(audio_arrays, audio_paths),
                    ))
        else:
            outputs = list(
                map(
                    self.process_single_example,
                    audio_arrays,
                    audio_paths,
                ))

        return outputs


def preprocess_batch(
        audio_arrays: Sequence[Array],
        audio_paths: Sequence[str],
        resampled_seq_length: Optional[int] = None) -> Sequence[AudioExample]:

    audio_examples = [AudioExample() for _ in range(len(audio_arrays))]

    for preprocessor in _PREPROCESSOR_LIST:
        preprocessed = preprocessor(audio_arrays, audio_paths)

        if preprocessor.mode == 'static':
            for ae, label in zip(audio_examples, preprocessed):
                if preprocessor.dtype == str:
                    ae.metadata[preprocessor.name] = str(label)
                else:
                    buffer = ae.buffers[preprocessor.name]
                    buffer.data = np.asarray(label).astype(
                        preprocessor.dtype).tobytes()
                    buffer.precision = DTYPE_TO_PRECISION[preprocessor.dtype]
                    buffer.metadata['mode'] = 'static'
                    buffer.shape.extend([1])

        elif preprocessor.mode == 'dynamic':
            for ae, array in zip(audio_examples, preprocessed):
                if preprocessor.resample and resampled_seq_length is not None:
                    array = interpolate(
                        array,
                        resampled_seq_length,
                        preprocessor.resampling_mode,
                    )
                buffer = ae.buffers[preprocessor.name]

                buffer.data = array.astype(preprocessor.dtype).tobytes()
                buffer.shape.extend(array.shape)
                buffer.precision = DTYPE_TO_PRECISION[preprocessor.dtype]
                buffer.metadata['mode'] = 'dynamic'

    return audio_examples


def register_preprocessor(preprocessor: Preprocessor):
    _PREPROCESSOR_LIST.append(preprocessor)


def get_preprocessors() -> Sequence[Preprocessor]:
    return _PREPROCESSOR_LIST


def set_default_gpu(device: torch.device):
    global _PREPROCESSOR_DEFAULT_GPU
    _PREPROCESSOR_DEFAULT_GPU = device


def get_default_gpu():
    return _PREPROCESSOR_DEFAULT_GPU


def set_sampling_rate(sampling_rate: int):
    global _SAMPLING_RATE
    _SAMPLING_RATE = sampling_rate


def get_sampling_rate():
    return _SAMPLING_RATE
