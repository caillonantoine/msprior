import cached_conv as cc
import pytest
import torch

from msprior import attention


@torch.enable_grad()
def test_causal():
    cc.use_cached_conv(False)
    seq_len = 128

    layer = attention.MultiHeadAlibiAttention(8)

    x = torch.randn(1, seq_len, 64, requires_grad=True)

    y = layer(x, x, x)
    y[:, seq_len // 2, :].mean().backward()
    grad = x.grad.data[0].sum(-1).reshape(-1)

    assert (grad[seq_len // 2 + 1:] == 0).all()
    assert y.shape == x.shape


@torch.no_grad()
@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
def test_inference(ratio):
    cc.use_cached_conv(False)
    no_cache = attention.MultiHeadAlibiAttention(8, ratio=ratio)

    cc.use_cached_conv(True)
    cache = attention.MultiHeadAlibiAttention(8, ratio=ratio)

    seq_len = 128

    q = torch.randn(1, seq_len, 64)
    k = torch.randn(1, seq_len // ratio, 64)
    v = torch.randn(1, seq_len // ratio, 64)

    pred_no_cache = no_cache(q, k, v)

    for i in range(2):
        pred_cache = []
        cache.reset()

        for i in range(seq_len):
            pred_cache.append(
                cache(q[:, i:i + 1], k[:, i // ratio:i // ratio + 1],
                      v[:, i // ratio:i // ratio + 1]))

        pred_cache = torch.cat(pred_cache, 1)

        assert pred_cache.shape == pred_no_cache.shape
        assert torch.allclose(pred_cache, pred_no_cache, rtol=1e-4, atol=1e-4)
