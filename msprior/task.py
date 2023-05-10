from random import randint
from typing import Dict, Tuple

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
