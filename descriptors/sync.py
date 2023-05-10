import librosa as li

from msprior.preprocessor import *


def beat_track_to_saw(beat_track: np.ndarray) -> np.ndarray:
    ramps = list(
        map(
            lambda segment: np.linspace(0, 1, segment[1] - segment[0]),
            zip(beat_track[:-1], beat_track[1:]),
        ))

    ramps = [np.zeros(beat_track[0])] + ramps
    return np.concatenate(ramps, -1)


class Sync(Preprocessor):

    def process_single_example(self, audio_array: Array,
                               audio_path: str) -> Union[Scalar, Array]:
        beat_track = li.beat.beat_track(
            y=audio_array,
            sr=self.sampling_rate,
            units='samples',
        )[1]

        if len(beat_track):
            saw = beat_track_to_saw(beat_track=beat_track)
        else:
            saw = None
        full_saw = np.zeros_like(audio_array)
        if saw is not None:
            full_saw[:len(saw)] = saw

        return full_saw


register_preprocessor(
    Sync(
        name='sync',
        mode='dynamic',
        dtype=np.float32,
        parallel=True,
    ))
