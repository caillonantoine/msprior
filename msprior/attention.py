import logging
import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import cached_conv as cc
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import msprior
from msprior import utils
from msprior import rwkv

TensorDict = Dict[str, torch.Tensor]


class MultiHeadAlibiAttention(nn.Module):

    def __init__(self,
                 n_head: int,
                 max_seq_len: int = 256,
                 ratio: int = 1) -> None:
        super().__init__()
        assert n_head >= 8, f'Alibi needs n_head > 8, got {n_head}'

        self._n_head = n_head
        self._ratio = ratio

        self._cached = cc.USE_BUFFER_CONV
        self._max_seq_len = max_seq_len // ratio

        self.register_cached_attention_bias()
        self.register_buffer("_keys_cache", torch.tensor([]))
        self.register_buffer("_values_cache", torch.tensor([]))
        self.register_buffer("_cache_length", torch.tensor(0))
        self.register_buffer("_relative_index", torch.tensor(0))

    def bias_attention(self, attention: torch.Tensor):
        q_len, k_len = attention.shape[-2:]
        bias = self._cached_attention_bias[..., -q_len:, -k_len:]
        return attention + bias[..., -q_len:, -k_len:]

    def register_cached_attention_bias(self):
        bias = torch.arange(self._max_seq_len)[None] * self._ratio
        bias = bias - torch.arange(self._max_seq_len * self._ratio)[:, None]
        bias = bias.float()
        bias.masked_fill_(bias > 0, -float("inf"))
        bias = bias[None]

        m = torch.ones(self._n_head) * 2**(-8 / self._n_head)
        m = torch.cumprod(m, -1)[:, None, None]
        bias = bias * m
        self.register_buffer('_cached_attention_bias', bias)

    @torch.jit.unused
    def init_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        self._keys_cache = torch.zeros(
            cc.MAX_BATCH_SIZE,
            k.shape[1],
            self._max_seq_len,
            k.shape[-1],
        ).type_as(k)
        self._values_cache = torch.zeros(
            cc.MAX_BATCH_SIZE,
            v.shape[1],
            self._max_seq_len,
            v.shape[-1],
        ).type_as(v)

    def update_cache(self, k: torch.Tensor, v: torch.Tensor):
        if not len(self._keys_cache) or not len(self._values_cache):
            self.init_kv_cache(k, v)

        input_length = k.shape[2]

        if input_length > self._max_seq_len:
            k = k[:k.shape[0], :, -self._max_seq_len:]
            v = v[:, :, -self._max_seq_len:]
            input_length = self._max_seq_len

        self._keys_cache = self._keys_cache.roll(-input_length, dims=2)
        self._keys_cache[:k.shape[0], :, -input_length:] = k

        self._values_cache = self._values_cache.roll(-input_length, dims=2)
        self._values_cache[:k.shape[0], :, -input_length:] = v

        if self._cache_length < self._max_seq_len:
            self._cache_length += input_length

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], self._n_head, -1)
        x = x.permute(0, 2, 1, 3)
        return x

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    def reset(self):
        self._cache_length.zero_()

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor] = None,
                v: Optional[torch.Tensor] = None) -> torch.Tensor:
        autoregressive = q.shape[1] == 1

        kv_available = k is not None or v is not None

        if autoregressive and self._cached:
            kv_available = kv_available and not self._relative_index

        q = self.split_heads(q)

        if kv_available:
            assert k is not None
            assert v is not None
            k = self.split_heads(k)
            v = self.split_heads(v)

        if self._cached:
            if kv_available:
                assert k is not None
                assert v is not None
                self.update_cache(k, v)

            k = self._keys_cache[:q.shape[0]]
            v = self._values_cache[:q.shape[0]]

        assert q is not None
        assert k is not None
        assert v is not None

        content_score = torch.einsum('bhtd,bhsd->bhts', q, k)
        attention = content_score / math.sqrt(q.shape[-1])

        # relative positional embedding
        if autoregressive and self._cached:
            bias_idx = -self._ratio + self._relative_index
            bias = self._cached_attention_bias[:, bias_idx, :]
            attention = attention + bias.unsqueeze(1)
        else:
            attention = self.bias_attention(attention)

        if self._cached and self._cache_length:
            attention[..., :-self._cache_length] = -float('inf')

        if self._cached:
            self._relative_index += q.shape[2]
            self._relative_index = self._relative_index % self._ratio

        attention = torch.softmax(attention, -1)

        out = torch.einsum('bhts,bhsd->bhtd', attention, v)
        out = self.gather_heads(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class GRULayer(nn.Module):

    def __init__(self, dropout_rate: float, model_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim,
            dropout=dropout_rate,
            num_layers=num_layers,
            batch_first=True,
        )
        self.cached = cc.USE_BUFFER_CONV
        self.register_buffer(
            "_state", torch.zeros(num_layers, cc.MAX_BATCH_SIZE, model_dim))

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is not None:
            raise ValueError(
                f"GRU layer only supported with decoder only mode.")
        if self.cached:
            batch_size = x.shape[0]
            x, state = self.gru(x, self._state[:, :batch_size])
            self._state[:, :batch_size].copy_(state)
        else:
            x, state = self.gru(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self,
                 attention_factory: Callable[[], MultiHeadAlibiAttention],
                 feed_forward_factory: Callable[[], FeedForward],
                 dropout_rate: float,
                 attend_encoder_out: bool,
                 model_dim: int,
                 head_dim: int,
                 num_heads: int,
                 encoder_out_ratio: Optional[int] = None) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(model_dim)
        self.q_linear = nn.Linear(model_dim, head_dim * num_heads)
        self.kv_linear = nn.Linear(model_dim, 2 * head_dim * num_heads)
        self.attention = attention_factory()
        self.post_attention_linear = nn.Linear(head_dim * num_heads, model_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

        if attend_encoder_out:
            assert encoder_out_ratio is not None
            self.encoder_attention_norm = nn.LayerNorm(model_dim)
            self.encoder_q_linear = nn.Linear(model_dim, head_dim * num_heads)
            self.encoder_kv_linear = nn.Linear(model_dim,
                                               2 * head_dim * num_heads)
            self.encoder_attention = attention_factory(ratio=encoder_out_ratio)
            self.encoder_post_attention_linear = nn.Linear(
                head_dim * num_heads, model_dim)
            self.encoder_attention_dropout = nn.Dropout(dropout_rate)

        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = feed_forward_factory()
        self.feed_forward_dropout = nn.Dropout(dropout_rate)

        self.attend_encoder_out = attend_encoder_out

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:

        # SELF ATTENTION
        x_res = x.clone()
        q = self.q_linear(x)

        k, v = self.kv_linear(x).chunk(2, -1)

        x = self.attention(q, k, v)
        x = self.post_attention_linear(x)
        x = self.attention_dropout(x)
        x = x + x_res
        x = self.attention_norm(x)

        # CROSS ATTENTION
        if self.attend_encoder_out:
            assert hasattr(self, "encoder_q_linear")
            assert y is not None

            x_res = x.clone()
            q = self.encoder_q_linear(x)

            k, v = self.encoder_kv_linear(y).chunk(2, -1)

            x = self.encoder_attention(q, k, v)
            x = self.encoder_post_attention_linear(x)
            x = self.encoder_attention_dropout(x)
            x = x + x_res
            x = self.encoder_attention_norm(x)

        # FEED FORWARD
        x_res = x.clone()
        x = self.feed_forward(x)
        x = self.feed_forward_dropout(x)
        x = x + x_res
        x = self.feed_forward_norm(x)

        return x


class ModulatedTransformerLayer(nn.Module):

    def __init__(self, attention_factory: Callable[[],
                                                   MultiHeadAlibiAttention],
                 feed_forward_factory: Callable[[], FeedForward],
                 dropout_rate: float, model_dim: int, head_dim: int,
                 num_heads: int, film: Callable[[], rwkv.Film]) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(model_dim)
        self.q_linear = nn.Linear(model_dim, head_dim * num_heads)
        self.kv_linear = nn.Linear(model_dim, 2 * head_dim * num_heads)
        self.attention = attention_factory()
        self.post_attention_linear = nn.Linear(head_dim * num_heads, model_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = feed_forward_factory()
        self.feed_forward_dropout = nn.Dropout(dropout_rate)

        self.film_norm = nn.LayerNorm(model_dim)
        self.film = film()

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert y is not None

        # SELF ATTENTION
        x_res = x.clone()
        x = self.attention_norm(x)
        q = self.q_linear(x)
        k, v = self.kv_linear(x).chunk(2, -1)
        x = self.attention(q, k, v)
        x = self.post_attention_linear(x)
        x = self.attention_dropout(x)
        x = x + x_res

        # FEED FORWARD
        x_res = x.clone()
        x = self.feed_forward_norm(x)
        x = self.feed_forward(x)
        x = self.feed_forward_dropout(x)
        x = x + x_res

        # MODULATION

        x_res = x.clone()
        x = self.film_norm(x)
        x = self.film(x, y)
        x = x + x_res

        return x


class MultivariateEmbedding(nn.Module):

    def __init__(self,
                 num_tokens: int,
                 num_features: int,
                 num_quantizers: int,
                 from_pretrained: Optional[str] = None) -> None:
        super().__init__()
        self.from_pretrained = from_pretrained

        self.embedder = nn.Embedding(num_quantizers * num_tokens, num_features)
        self.proj = None

        if from_pretrained is not None:
            model = torch.jit.load(from_pretrained).eval()
            if hasattr(model.encoder, "rvq"):
                embeds = []
                for n, p in model.encoder.rvq.named_buffers():
                    if n[-14:] == "codebook.embed":
                        embeds.append(p)
                embeds = torch.cat(embeds, 0)

                self.embedder = nn.Embedding(num_quantizers * num_tokens,
                                             embeds.shape[-1])
                self.embedder.weight.data.copy_(embeds)
                self.proj = nn.Linear(embeds.shape[-1], num_features)
            else:
                logging.warn(
                    "pretrained_embedding is only compatible with discrete rave models, skiping"
                )

        self.num_quantizers = num_quantizers
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor,
                sum_over_quantizers: bool) -> torch.Tensor:
        if sum_over_quantizers:
            x = x + torch.arange(x.shape[-1]).type_as(x) * self.num_tokens

        x = self.embedder(x.long())

        if sum_over_quantizers:
            x = x.sum(-2)

        if self.from_pretrained is not None:
            x = x.detach()
            x = self.proj(x)

        return x


class Embedding(nn.Embedding):

    def forward(self,
                input: torch.Tensor,
                sum_over_quantizers: Optional[bool] = None) -> torch.Tensor:
        input = input.long()

        return nn.functional.embedding(input, self.weight, self.padding_idx,
                                       self.max_norm, self.norm_type,
                                       self.scale_grad_by_freq, self.sparse)


class LogitsProjection(nn.Module):

    def __init__(self, dim: int, num_tokens: int, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Linear(dim, num_tokens))

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.net(inputs)


class FeatureEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.permute(0, 2, 1))
        return self.net(x.permute(0, 2, 1))


class IndividualPredictor(nn.Module):

    def __init__(self, dim: int, num_tokens: int, first_quantizer: bool,
                 dropout: float) -> None:
        super().__init__()
        self.predictor = nn.Sequential(nn.GELU(), nn.Linear(dim, num_tokens))
        self.dropout = nn.Dropout(dropout)

        if not first_quantizer:
            self.film = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                      nn.Linear(dim, 2 * dim))
        else:
            self.film = None

    def forward(self,
                context: torch.Tensor,
                previous_embedding: Optional[torch.Tensor] = None):

        if previous_embedding is not None and self.film is not None:
            film = self.film(self.dropout(previous_embedding))
            scale, bias = torch.chunk(film, 2, -1)
            context = context * scale + bias

        logits = self.predictor(context)

        return logits


class MultivariatePredictor(nn.Module):

    def __init__(
        self,
        dim: int,
        num_quantizers: int,
        num_tokens: int,
        dropout_rate: float,
        shared_embedder: MultivariateEmbedding,
    ) -> None:
        super().__init__()
        self.embedder = shared_embedder

        self.predictors = nn.ModuleList([
            IndividualPredictor(dim, num_tokens, idx == 0, dropout_rate)
            for idx in range(num_quantizers)
        ])
        self.num_tokens = num_tokens
        self.num_quantizers = num_quantizers

    def _forward(
        self,
        context: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_logits = []
        samples = []

        embedded_token = torch.zeros_like(context)
        for quant_id, predictor in enumerate(self.predictors):
            logits = predictor(context, embedded_token)
            all_logits.append(logits)

            if targets is not None:
                current_token = targets[..., quant_id]
            else:
                current_token = utils.sample_from_logits(logits, temperature)

            samples.append(current_token)

            if quant_id != self.num_quantizers - 1:
                embedded_token += self.embedder(
                    current_token.long() + quant_id * self.num_tokens,
                    sum_over_quantizers=False,
                )

        all_logits = torch.stack(all_logits, -2)
        samples = torch.stack(samples, -1)

        dist = torch.log_softmax(all_logits, -1)
        entropy = -(dist * dist.exp()).sum(-1)
        perplexity = entropy.exp().mean(-1)

        return all_logits, samples, perplexity

    def forward(self,
                context: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> torch.Tensor:
        return self._forward(context, targets, temperature)[0]

    def sample(self, x: torch.Tensor,
               temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward(x, temperature=temperature)[1:]


class Encoder(nn.Module):

    def __init__(
        self,
        embedder: Callable[[], nn.Module],
        layer_factory: Callable[[], TransformerLayer],
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedder = embedder()

        net = []
        for _ in range(num_layers):
            net.append(layer_factory())
        self.net = nn.ModuleList(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)

        for layer in self.net:
            x = layer(x)

        return x


class Decoder(nn.Module):

    def __init__(self, embedder: Callable[[], MultivariateEmbedding],
                 layer_factory: Callable[[], TransformerLayer],
                 predictor: Callable[[], MultivariatePredictor],
                 num_layers: int) -> None:
        super().__init__()
        self.embedder = embedder()

        net = []
        for _ in range(num_layers):
            net.append(layer_factory())
        self.net = nn.ModuleList(net)
        self.streaming = cc.USE_BUFFER_CONV

        self.predictor = predictor(shared_embedder=self.embedder)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedder(x, sum_over_quantizers=True)

        for layer in self.net:
            x = layer(x, y)

        return self.predictor(x, targets)

    def sample(self,
               x: torch.Tensor,
               y: Optional[torch.Tensor] = None,
               temperature: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedder(x, sum_over_quantizers=True)

        for layer in self.net:
            x = layer(x, y)

        return self.predictor.sample(x, temperature=temperature)


@gin.configurable
class Prior(pl.LightningModule):

    def __init__(
        self,
        decoder_factory: Callable[[], Decoder],
        encoder_factory: Optional[Callable[[], Encoder]] = None,
        inputs_preprocessing: Optional[Callable[[TensorDict],
                                                TensorDict]] = None,
    ) -> None:
        super().__init__()

        self.encoder = None
        if encoder_factory is not None:
            self.encoder = encoder_factory()
        self.decoder = decoder_factory()

        self.inputs_preprocessing = inputs_preprocessing

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            assert y is not None
            encoder_out = self.encoder(y)
        else:
            encoder_out = None

        return self.decoder(x, encoder_out, targets)

    def sample(self,
               x: torch.Tensor,
               encoder_out: Optional[torch.Tensor] = None,
               temperature: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            assert encoder_out is not None

        return self.decoder.sample(
            x,
            encoder_out,
            temperature=temperature,
        )

    def loss(self, inputs: TensorDict) -> torch.Tensor:
        if self.inputs_preprocessing is not None:
            inputs = self.inputs_preprocessing(inputs)

        logits = self.forward(
            inputs["decoder_inputs"],
            inputs["encoder_inputs"],
            inputs["decoder_targets"],
        )

        targets_one_hot = nn.functional.one_hot(
            torch.clamp(inputs["decoder_targets"].long(), 0),
            logits.shape[-1],
        ).float()

        logits = torch.log_softmax(logits, -1)

        cross_entropy = -(logits * targets_one_hot).sum(-1)

        return cross_entropy.mean(), logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.loss(batch)

        self.log('cross_entropy', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)

        self.log('val_cross_entropy', loss)

        accuracies = self.accuracy(logits, batch["decoder_targets"])
        for topk, acc in accuracies:
            self.log(f'val_acc_top_{topk}', acc)

    @gin.configurable
    def configure_optimizers(
            self, optimizer_cls: Callable[[], torch.optim.Optimizer],
            scheduler_cls: Callable[[],
                                    torch.optim.lr_scheduler._LRScheduler]):
        optimizer = optimizer_cls(self.parameters())
        scheduler = scheduler_cls(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def accuracy(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> Sequence[Tuple[float, float]]:
        prediction = prediction.cpu()
        target = target.cpu()

        top_10 = torch.topk(prediction, 10, -1).indices
        accuracies = (target[..., None] == top_10).long()
        k_values = [1, 3, 5, 10]
        k_accuracy = []
        for k in k_values:
            current = (accuracies[..., :k].sum(-1) != 0).float()
            k_accuracy.append(current.mean())
        return list(zip(k_values, k_accuracy))

    def on_fit_start(self):
        tb = self.logger.experiment
        config = gin.config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)
