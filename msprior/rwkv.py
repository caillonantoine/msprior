# Adapted from https://github.com/BlinkDL/RWKV-LM

import math
import os
from typing import Callable, Optional

import cached_conv as cc
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import logging

T_MAX = 256

wkv_cuda = None


class WKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C),
                        device='cuda',
                        memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)


def WKV_Attention(B, T, C, w, u, k, v):
    global wkv_cuda
    if wkv_cuda is None:
        logging.info("Building custom CUDA kernels")
        wkv_cuda = load(
            name=f"wkv_{T_MAX}",
            sources=[
                os.path.join(os.path.dirname(__file__), "cuda/wkv_op.cpp"),
                os.path.join(os.path.dirname(__file__), "cuda/wkv_cuda.cu"),
            ],
            verbose=False,
            extra_cuda_cflags=[
                "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
                "--use_fast_math", "-O3", "-Xptxas -O3",
                "--extra-device-vectorization", f"-DTmax={T_MAX}"
            ],
        )
    return WKV.apply(B, T, C, w, u, k, v)


class RWKV_TimeMix(nn.Module):

    def __init__(self, layer_id: int, ctx_len: int, dim: int, num_layers: int):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.dim = dim

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (num_layers - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers))  # 1 to ~0

            # time_decay init
            decay_speed = torch.ones(dim)
            for h in range(dim):
                decay_speed[h] = -5 + 8 * (h / (dim - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # time_first init
            time_first = (torch.tensor([(i + 1) % 3 - 1
                                        for i in range(dim)]) * 0.5)
            self.time_first = nn.Parameter(
                torch.ones(dim) * math.log(0.3) + time_first)

            # time_mix init
            x = torch.ones(1, 1, dim)
            for i in range(dim):
                x[0, 0, i] = i / dim
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

        self.output = nn.Linear(dim, dim, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        rwkv = sr * WKV_Attention(
            B,
            T,
            C,
            self.time_decay,
            self.time_first,
            k,
            v,
        )
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):

    def __init__(self, layer_id: int, num_layers: int, dim: int):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers))  # 1 to ~0

            x = torch.ones(1, 1, dim)
            for i in range(dim):
                x[0, 0, i] = i / dim

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * dim
        self.key = nn.Linear(dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_sz, dim, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class IncrementalTimeMix(nn.Module):

    def __init__(self, layer_id: int, ctx_len: int, dim: int, num_layers: int):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.dim = dim

        self.time_decay = nn.Parameter(torch.empty(dim))
        self.time_first = nn.Parameter(torch.empty(dim))
        self.time_mix_k = nn.Parameter(torch.empty(dim))
        self.time_mix_v = nn.Parameter(torch.empty(dim))
        self.time_mix_r = nn.Parameter(torch.empty(dim))

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

        self.output = nn.Linear(dim, dim, bias=False)

        self.register_buffer("last_x", torch.zeros(cc.MAX_BATCH_SIZE, 1, dim))
        self.register_buffer("last_num", torch.zeros(cc.MAX_BATCH_SIZE, 1,
                                                     dim))
        self.register_buffer("last_den", torch.zeros(cc.MAX_BATCH_SIZE, 1,
                                                     dim))

    def interpolate(self, x: torch.Tensor,
                    factor: torch.Tensor) -> torch.Tensor:
        x = x * factor + self.last_x[:x.shape[0]] * (1 - factor)
        return x

    def update_cache(self, x: torch.Tensor, num: torch.Tensor,
                     den: torch.Tensor) -> None:
        batch_size = x.shape[0]

        self.last_x[:batch_size].copy_(x)
        self.last_num[:batch_size].copy_(num)
        self.last_den[:batch_size].copy_(den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        k = self.key(self.interpolate(x, self.time_mix_k))
        v = self.value(self.interpolate(x, self.time_mix_v))
        r = self.receptance(self.interpolate(x, self.time_mix_r))

        exp_k = torch.exp(self.time_first + k)
        wkv = (self.last_num[:batch_size] +
               exp_k * v) / (self.last_den[:batch_size] + exp_k)
        rwkv = torch.sigmoid(r) * wkv

        decay = torch.exp(-torch.exp(self.time_decay))
        exp_k = torch.exp(k)
        num = decay * self.last_num[:batch_size] + exp_k * v
        den = decay * self.last_den[:batch_size] + exp_k

        out = self.output(rwkv)

        self.update_cache(x, num, den)
        return out


class IncrementalChannelMix(nn.Module):

    def __init__(self, layer_id: int, num_layers: int, dim: int):
        super().__init__()
        self.layer_id = layer_id

        self.time_mix_k = nn.Parameter(torch.empty(dim))
        self.time_mix_r = nn.Parameter(torch.empty(dim))

        hidden_sz = 4 * dim
        self.key = nn.Linear(dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_sz, dim, bias=False)

        self.register_buffer("last_x", torch.zeros(cc.MAX_BATCH_SIZE, 1, dim))

    def interpolate(self, x: torch.Tensor,
                    factor: torch.Tensor) -> torch.Tensor:
        x = x * factor + self.last_x[:x.shape[0]] * (1 - factor)
        return x

    def update_cache(self, x: torch.Tensor) -> None:
        batch_size = x.shape[0]

        self.last_x[:batch_size].copy_(x)

    def forward(self, x: torch.Tensor):
        k = self.key(self.interpolate(x, self.time_mix_k))
        r = self.receptance(self.interpolate(x, self.time_mix_r))

        vk = self.value(torch.relu(k).pow(2))

        out = torch.sigmoid(r) * vk

        self.update_cache(x)
        return out


class Block(nn.Module):

    def __init__(self, layer_id: int, dim: int,
                 time_mixer: Callable[[int], RWKV_TimeMix],
                 channel_mixer: Callable[[int], RWKV_ChannelMix]) -> None:
        super().__init__()
        self.t_mixer = nn.Sequential(nn.LayerNorm(dim), time_mixer(layer_id))
        self.c_mixer = nn.Sequential(nn.LayerNorm(dim),
                                     channel_mixer(layer_id))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_mixer(x) + x
        x = self.c_mixer(x) + x
        return x


class Film(nn.Module):

    def __init__(self, in_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, scale = self.net(y).chunk(2, -1)
        return x * scale + mean


class RWKV(nn.Module):

    def __init__(self,
                 block: Callable[[int], Block],
                 num_layers: int,
                 dim: int,
                 film: Optional[Callable[[], Film]] = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([block(i) for i in range(num_layers)])
        self.films = None
        if film is not None:
            self.films = nn.ModuleList([film() for _ in range(num_layers)])

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cc.USE_BUFFER_CONV and x.shape[1] != 1:
            out = []
            if self.films is not None:
                assert y is not None
                for _x, _y in zip(
                        x.chunk(x.shape[1], 1),
                        y.chunk(y.shape[1], 1),
                ):
                    _x = self.norm(_x)
                    for block, film in zip(self.blocks, self.films):
                        _x = block(_x)
                        _x = film(_x, _y)
                    out.append(_x)
            else:
                for _x in x.chunk(x.shape[1], 1):
                    _x = self.norm(_x)
                    for block in self.blocks:
                        _x = block(_x)
                    out.append(_x)
            x = torch.cat(out, 1)
        else:
            x = self.norm(x)
            if self.films is not None:
                assert y is not None
                for block, film in zip(self.blocks, self.films):
                    x = block(x)
                    x = film(x, y)
            else:
                for block in self.blocks:
                    x = block(x)
        return x


def TimeMixer(layer_id: int, ctx_len: int, dim: int, num_layers: int):
    if cc.USE_BUFFER_CONV:
        return IncrementalTimeMix(layer_id, ctx_len, dim, num_layers)
    else:
        return RWKV_TimeMix(layer_id, ctx_len, dim, num_layers)


def ChannelMixer(layer_id: int, num_layers: int, dim: int):
    if cc.USE_BUFFER_CONV:
        return IncrementalChannelMix(layer_id, num_layers, dim)
    else:
        return RWKV_ChannelMix(layer_id, num_layers, dim)
