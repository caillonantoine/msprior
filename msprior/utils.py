import functools
import multiprocessing
import os
import pathlib
import subprocess
from itertools import repeat
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import gin
import numpy as np
import torch
import torch.nn as nn

# GIN UTILS

TensorDict = Dict[str, torch.Tensor]


def get_feed_forward_size(model_dim):
    return 4 * model_dim


@gin.configurable
def build_warmed_exponential_lr_scheduler(
        optim: torch.optim.Optimizer, start_factor: float, peak_iteration: int,
        decay_factor: float) -> torch.optim.lr_scheduler._LRScheduler:
    linear = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=start_factor,
        end_factor=1.,
        total_iters=peak_iteration,
    )
    exp = torch.optim.lr_scheduler.ExponentialLR(
        optim,
        gamma=decay_factor,
    )
    return torch.optim.lr_scheduler.SequentialLR(optim, [linear, exp],
                                                 milestones=[peak_iteration])


def total_num_tokens(num_quantizers: int, num_tokens: int,
                     num_reserved_tokens: int):
    return num_quantizers * (num_tokens + num_reserved_tokens)


def per_quantizer_num_tokens(num_tokens: int, num_reserved_tokens: int):
    return num_tokens + num_reserved_tokens


def get_num_quantizers(rave_path: str,
                       quantizer_truncation: Optional[int] = None) -> int:
    if quantizer_truncation is not None:
        return quantizer_truncation
    rave = torch.jit.load(rave_path).eval()
    return rave.encode_params[2].item()


# MODEL UTILS


@torch.no_grad()
def perplexity(pred: torch.Tensor):
    pred = torch.log_softmax(pred, -1)
    entropy = -(pred * torch.exp(pred))
    entropy.masked_fill_(torch.isnan(entropy), 0)
    perplexity = torch.exp(entropy.sum(-1)).mean()
    return perplexity


def load_run(run: Optional[str] = None) -> Tuple[str, nn.Module]:

    @gin.configurable
    def get_model(model):
        return model

    print(f'loading run {run}')
    if run is None: return None

    config = os.path.join(run, 'config.gin')
    gin.clear_config()
    gin.parse_config_file(config)
    gin.parse_config(r'get_model.model = %FULL_MODEL')
    model = get_model()

    try:
        ckpt = next(iter(pathlib.Path(run).rglob('best*')))
        ckpt = torch.load(ckpt, map_location="cpu")["state_dict"]
        state = model.state_dict()
        state.update(ckpt)
        model.load_state_dict(state)
    except:
        print(
            f'No compatible ckpt found in {run}, model is left randomly initialised...'
        )
        model = get_model()

    model.eval()

    return run, model


@torch.no_grad()
def pretrained_embedder_init(pretrained_path: str, num_reserved_tokens: int):
    model = torch.jit.load(pretrained_path).eval()
    codebook = model.quantizer.embed
    additional_tokens = torch.randn(
        codebook.shape[0],
        num_reserved_tokens,
        codebook.shape[-1],
    )
    codebook = torch.cat([additional_tokens, codebook], 1)
    codebook = codebook.reshape(-1, codebook.shape[-1])

    embedding = nn.Embedding(*codebook.shape)
    embedding.weight.data.copy_(codebook)
    return embedding


# PREPROCESSING UTILS


def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


def load_audio_chunk(
        path: str,
        n_signal: int,
        sr: int,
        silenceremove: Optional[int] = None) -> Iterable[np.ndarray]:

    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'panic',
        '-i',
        path,
        '-ac',
        '1',
        '-ar',
        str(sr),
    ]

    if silenceremove is not None:
        ffmpeg_cmd.extend(["-af", f"silenceremove=1:0:-{silenceremove}dB"])

    ffmpeg_cmd.extend(['-f', 's16le', '-'])
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
    )
    chunk = process.stdout.read(n_signal * 2)
    while len(chunk) == n_signal * 2:
        chunk = np.frombuffer(chunk, np.int16).astype(np.float32) / 2**15
        yield chunk, path
        chunk = process.stdout.read(n_signal * 2)

    process.stdout.close()


def batch(iterator: Iterable, batch_size: int):
    batch = []
    for elm in iterator:
        batch.append(elm)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch):
        yield batch


def tuple_as_args(fun):

    def wrapper(tuple_arg):
        return fun(*tuple_arg)

    return wrapper


def flatmap(
    pool: multiprocessing.Pool,
    func: Callable,
    iterable: Iterable,
    queue_size: int,
    chunksize=None,
):
    queue = multiprocessing.Manager().Queue(maxsize=queue_size)
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


# PREPROCESSING UTILS


def build_target(inputs: Dict[str, torch.Tensor], input_key: str):
    target_tensor = inputs[input_key].roll(-1, 1)
    target_tensor[:, -1] = -1
    inputs['targets'] = target_tensor


def look_ahead(inputs: Dict[str, torch.Tensor], key: str):
    inputs[key] = inputs[key].roll(-1, 1)
    inputs[key][:, -1] = inputs[key][:, -2]


def compose(functions: Sequence[Callable[[Dict[str, torch.Tensor]], None]]):

    def composed_function(inputs: Dict[str, torch.Tensor]):
        for f in functions:
            f(inputs)

    return composed_function


# OTHERS

# def sample_from_logits(logits: torch.Tensor,
#                        temperature: float = 0.) -> torch.Tensor:
#     assert temperature >= 0
#     dist = torch.softmax(logits / (temperature + 1e-15), -1)
#     dist = dist.reshape(-1, dist.shape[-1])
#     samples = torch.multinomial(dist, 1).reshape_as(logits[..., 0])
#     return samples


def sample_from_logits(logits: torch.Tensor,
                       temperature: float = 0.) -> torch.Tensor:
    if temperature != 0:
        logits = torch.log_softmax(logits / temperature, -1)
        logits -= torch.empty_like(logits).exponential_().log()

    samples = torch.argmax(logits, -1)
    return samples


# CONTINUOUS FEATURE RECONSTRUCTION


def semantic_vae_processing(
        vae_path: Optional[str] = None
) -> Callable[[torch.Tensor], torch.Tensor]:

    if vae_path is None:
        return lambda x: x

    vae = torch.jit.load(vae_path).eval()
    state = list(
        map(lambda nv: nv[1],
            filter(
                lambda nv: "_state" in nv[0],
                vae.named_buffers(),
            )))

    def reset():
        for elm in state:
            elm.zero_()

    @torch.no_grad()
    def vae_processing(inputs: TensorDict) -> TensorDict:
        reset()
        vae.to(inputs["encoder_inputs"].device)
        reconstruction = vae.forward(
            inputs["encoder_inputs"],
            context=inputs["decoder_inputs"],
        )

        inputs["encoder_inputs"] = reconstruction

        return inputs

    return vae_processing
