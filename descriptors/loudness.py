import dataclasses

import librosa as li
import numpy as np
import torch

from msprior.preprocessor import *

torch.set_grad_enabled(False)


@dataclasses.dataclass
class Loudness(Preprocessor):
    n_fft: int = 4096
    hop_length: int = 1024

    def __post_init__(self):
        super().__post_init__()
        a_weight = li.A_weighting(
            np.linspace(
                1,
                self.sampling_rate / 2,
                self.n_fft // 2 + 1,
            ))

        self.a_weight = torch.from_numpy(a_weight).float().to(self.device)
        self.a_weight = (10**self.a_weight).unsqueeze(-1)

    def __call__(self, audio_arrays: Sequence[Array],
                 audio_paths: Sequence[str]) -> Sequence[Union[Scalar, Array]]:
        audio = np.stack(audio_arrays, axis=0)
        audio = torch.from_numpy(audio).float().to(self.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            return_complex=True,
        ).abs()
        stft = stft * self.a_weight
        rms = torch.log1p(stft).mean(1)

        return rms.cpu().numpy()


register_preprocessor(
    Loudness(
        name='loudness',
        mode='dynamic',
        dtype=np.float32,
        hop_length=2**13,
        n_fft=2**15,
        resampling_mode='linear',
    ))
