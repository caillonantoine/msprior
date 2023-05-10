import numpy as np

from msprior.preprocessor import *


class WaveformPassThrough(Preprocessor):

    def __call__(self, audio_arrays: Sequence[Array],
                 audio_paths: Sequence[str]) -> Sequence[Union[Scalar, Array]]:
        audio_arrays = np.stack(audio_arrays, 0)
        audio_arrays = np.clip(audio_arrays, -1, 1)
        audio_arrays = np.round(audio_arrays * (2**15 - 1))
        return audio_arrays.astype(np.int16)


register_preprocessor(
    WaveformPassThrough(
        name='waveform',
        mode='dynamic',
        dtype=np.int16,
        resample=False,
    ))
