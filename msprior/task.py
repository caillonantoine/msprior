from random import randint
from typing import Dict

import numpy as np
import torch

TensorDict = Dict[str, torch.Tensor]


def decoder_only_rave(inputs: TensorDict, seq_len: int,
                      decoder_key: str) -> TensorDict:
    start = randint(0, inputs[decoder_key].shape[0] - seq_len - 1)
    end = start + seq_len

    input_tokens = inputs[decoder_key][start:end].copy()
    target_tokens = inputs[decoder_key][start + 1:end + 1].copy()

    return {
        "encoder_inputs": torch.Tensor([]),
        "decoder_inputs": input_tokens,
        "decoder_targets": target_tokens,
    }


def decoder_only_rave_flattened(inputs: TensorDict, seq_len: int,
                                decoder_key: str,
                                num_tokens: int) -> TensorDict:
    num_q = inputs[decoder_key].shape[-1]

    inputs = inputs[decoder_key] + np.arange(num_q) * num_tokens
    inputs = inputs.rehspae(-1)
    start = randint(0, inputs.shape[0] - seq_len - 1)
    if start % num_q:
        start -= start % num_q
    end = start + seq_len

    input_tokens = inputs[start:end].copy()
    target_tokens = inputs[start + 1:end + 1].copy()

    return {
        "encoder_inputs": torch.Tensor([]),
        "decoder_inputs": input_tokens,
        "decoder_targets": target_tokens,
    }


def encoder_decoder_semantic_token(inputs: TensorDict, seq_len: int,
                                   encoder_key: str,
                                   decoder_key: str) -> TensorDict:
    start = randint(0, inputs[decoder_key].shape[0] - seq_len - 1)
    ratio = inputs[decoder_key].shape[0] // inputs[encoder_key].shape[0]
    start = start - (start % ratio)
    end = start + seq_len

    input_tokens = inputs[decoder_key][start:end].copy()
    target_tokens = inputs[decoder_key][start + 1:end + 1].copy()

    semantic_indices = inputs[encoder_key].copy()
    semantic_indices = semantic_indices[start // ratio:end // ratio]

    return {
        "encoder_inputs": semantic_indices,
        "decoder_inputs": input_tokens,
        "decoder_targets": target_tokens,
    }


def encoder_decoder_semantic_features(inputs: TensorDict, seq_len: int,
                                      encoder_key: str,
                                      decoder_key: str) -> TensorDict:
    start = randint(0, inputs[decoder_key].shape[0] - seq_len - 1)
    ratio = inputs[decoder_key].shape[0] // inputs[encoder_key].shape[0]
    start = start - (start % ratio)
    end = start + seq_len

    input_tokens = inputs[decoder_key][start:end].copy()
    target_tokens = inputs[decoder_key][start + 1:end + 1].copy()

    semantic_features = inputs[encoder_key].copy()
    semantic_features = semantic_features[start // ratio:end // ratio]

    return {
        "encoder_inputs": semantic_features,
        "decoder_inputs": input_tokens,
        "decoder_targets": target_tokens,
    }


def conditioned_rwkv(inputs: TensorDict, seq_len: int, main_key: str,
                     condition_key: str) -> TensorDict:
    main_seq = inputs[main_key]
    condition_seq = inputs[condition_key]
    ratio = main_seq.shape[0] // condition_seq.shape[0]

    if ratio < 1:
        raise ValueError(
            f"time ratio between main and condition should be >= 1, got {ratio}"
        )

    condition_seq = np.repeat(condition_seq, ratio, axis=0)

    start = randint(0, main_seq.shape[0] - seq_len - 1)
    end = start + seq_len

    return {
        "encoder_inputs": condition_seq[start + 1:end + 1],
        "decoder_inputs": main_seq[start:end],
        "decoder_targets": main_seq[start + 1:end + 1],
    }
