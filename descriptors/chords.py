import dataclasses

import numpy as np
from madmom.audio import SignalProcessor
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

from msprior.preprocessor import *


@dataclasses.dataclass
class Chords(Preprocessor):

    def __post_init__(self):
        super().__post_init__()
        self.signal_proc = SignalProcessor(sample_rate=self.sampling_rate)
        self.deep_chroma = DeepChromaProcessor()
        self.chords_detector = DeepChromaChordRecognitionProcessor()
        self.keys = [
            'N', 'A:maj', 'A#:maj', 'B:maj', 'C:maj', 'C#:maj', 'D:maj',
            'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:min',
            'A#:min', 'B:min', 'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min',
            'F:min', 'F#:min', 'G:min', 'G#:min'
        ]

    def process_single_example(self, audio_array: Array,
                               audio_path: str) -> Union[Scalar, Array]:
        out = np.zeros_like(audio_array)
        audio_array = self.signal_proc(audio_array)
        chords = self.chords_detector(self.deep_chroma(audio_array))

        for start, stop, key in chords:
            start = int(self.sampling_rate * start)
            stop = int(self.sampling_rate * stop)

            key_idx = self.keys.index(key)
            out[start:stop] = key_idx

        return out


register_preprocessor(
    Chords(
        name='chords',
        mode='dynamic',
        dtype=np.int16,
        parallel=True,
    ))
