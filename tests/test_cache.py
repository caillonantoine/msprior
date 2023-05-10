import cached_conv as cc
import gin
import pytest
import torch

from msprior.attention import Prior
from msprior.scripted import ScriptedPrior

torch.set_grad_enabled(False)
torch.random.manual_seed(42)
import msprior


def load_configuration(path, overrides=[]):

    def with_config_decorator(fun):

        def wrapper(*args, **kwargs):
            gin.clear_config()
            gin.parse_config_file(path)
            for override in overrides:
                gin.parse_config(override)
            out = fun(*args, **kwargs)
            return out

        return wrapper

    return with_config_decorator


@load_configuration(
    path="decoder_only.gin",
    overrides=["MODEL_DIM = 16", "HEAD_DIM = 2", "NUM_LAYERS = 2"],
)
def test_decoder_only():
    cc.use_cached_conv(False)
    regular = Prior().eval()

    cc.use_cached_conv(True)
    streaming = Prior().eval()

    for p1, p2 in zip(regular.parameters(), streaming.parameters()):
        p2.data.copy_(p1.data)

    input_params = gin.get_bindings("attention.MultivariateEmbedding")
    inputs = torch.randint(
        0,
        input_params["num_tokens"],
        (1, 128, input_params["num_quantizers"]),
    )

    outputs_regular = regular(inputs, targets=inputs)

    num_chunks = 4
    outputs_streaming = []
    for inputs in inputs.chunk(num_chunks, 1):
        outputs_streaming.append(streaming(inputs, targets=inputs))
    outputs_streaming = torch.cat(outputs_streaming, 1)

    assert torch.allclose(outputs_regular, outputs_streaming, 1e-4, 1e-4)


@load_configuration(
    path="encoder_decoder.gin",
    overrides=["MODEL_DIM = 16", "HEAD_DIM = 2", "NUM_LAYERS = 2"],
)
def test_encoder_decoder():
    cc.use_cached_conv(False)
    regular = Prior().eval()

    cc.use_cached_conv(True)
    streaming = Prior().eval()

    for p1, p2 in zip(regular.parameters(), streaming.parameters()):
        p2.data.copy_(p1.data)

    input_params = gin.get_bindings("attention.MultivariateEmbedding")
    inputs = torch.randint(
        0,
        input_params["num_tokens"],
        (1, 128, input_params["num_quantizers"]),
    )

    semantic_params = gin.get_bindings("encoder/attention.Embedding")
    semantic = torch.randint(
        0,
        semantic_params["num_embeddings"],
        (1, 32),
    )

    outputs_regular = regular(inputs, semantic, targets=inputs)

    num_chunks = 4
    outputs_streaming = []
    for inputs, semantic in zip(inputs.chunk(num_chunks, 1),
                                semantic.chunk(num_chunks, 1)):
        outputs_streaming.append(streaming(inputs, semantic, targets=inputs))
    outputs_streaming = torch.cat(outputs_streaming, 1)

    assert torch.allclose(outputs_regular, outputs_streaming, 1e-4, 1e-4)


@load_configuration(
    path="decoder_only.gin",
    overrides=["MODEL_DIM = 16", "HEAD_DIM = 2", "NUM_LAYERS = 2"],
)
def test_decoder_only_nn_tilde():
    cc.use_cached_conv(False)
    regular = Prior().eval()

    cc.use_cached_conv(True)
    streaming = ScriptedPrior().eval()

    for p1, p2 in zip(regular.parameters(), streaming.parameters()):
        p2.data.copy_(p1.data)

    input_params = gin.get_bindings("attention.MultivariateEmbedding")
    inputs = torch.randint(
        0,
        input_params["num_tokens"],
        (1, 1, input_params["num_quantizers"]),
    )

    streaming.set_listen(False)
    streaming.set_reset(True)
    streaming.set_temperature(0)

    # AUTOREGRESSIVE SAMPLE USING REGULAR MODEL
    regular_samples = [inputs]
    for _ in range(4):
        regular_samples.append(
            regular.sample(
                torch.cat(regular_samples, 1),
                temperature=0,
            )[0][:, -1:])
    regular_samples = torch.cat(regular_samples, 1)

    # AUTOREGRESSIVE SAMPLE USING NN_TILDE
    streaming.previous_step[:1].copy_(inputs)
    out = streaming(torch.zeros(1, 16, 4))[:, :-1]
    streaming_samples = torch.cat([inputs, out.permute(0, 2, 1)], 1).long()

    assert torch.allclose(regular_samples, streaming_samples)


@load_configuration(
    path="encoder_decoder.gin",
    overrides=["MODEL_DIM = 8", "HEAD_DIM = 2", "NUM_LAYERS = 1"],
)
def test_encoder_decoder_nn_tilde():
    cc.use_cached_conv(False)
    regular = Prior().eval()

    cc.use_cached_conv(True)
    streaming = ScriptedPrior().eval()

    for p1, p2 in zip(regular.parameters(), streaming.parameters()):
        p2.data.copy_(p1.data)

    input_params = gin.get_bindings("attention.MultivariateEmbedding")
    inputs = torch.randint(
        0,
        input_params["num_tokens"],
        (1, 1, input_params["num_quantizers"]),
    )

    semantic_params = gin.get_bindings("encoder/attention.Embedding")
    semantic = torch.randint(
        0,
        semantic_params["num_embeddings"],
        (1, 16),
    )

    # AUTOREGRESSIVE SAMPLE USING REGULAR MODEL
    regular_samples = [inputs]

    regular_encoder_out = regular.encoder(semantic)

    for i in range(64):
        encoder_out_in = regular_encoder_out[:, :i // 4 + 1]
        codes_in = torch.cat(regular_samples, 1)
        sample, _ = regular.sample(
            codes_in,
            encoder_out_in,
            temperature=0,
        )
        regular_samples.append(sample[:, -1:])

    regular_samples = torch.cat(regular_samples, 1)

    # AUTOREGRESSIVE SAMPLE USING NN_TILDE
    streaming.set_listen(False)
    streaming.set_reset(True)
    streaming.set_temperature(0)
    streaming.previous_step[:1].copy_(inputs)

    semantic = semantic.repeat_interleave(4, -1)
    streaming_inputs = torch.cat([
        torch.zeros(1, 16, 64),
        semantic[:, :64][:, None],
    ], 1)
    out = streaming(streaming_inputs)[:, :-1]
    streaming_samples = torch.cat([inputs, out.permute(0, 2, 1)], 1).long()

    assert torch.allclose(regular_samples, streaming_samples)
