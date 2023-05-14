import math
import os
import pathlib
from typing import Optional

import cached_conv as cc
import gin
import nn_tilde
import torch

from .attention import Embedding, FeatureEmbedding, Prior


class ScriptedPrior(nn_tilde.Module):

    def __init__(
        self,
        run: Optional[str] = None,
        temporal_ratio: int = 1024,
        from_continuous: bool = False,
        vae_path: Optional[str] = None,
        initial_listen: bool = True,
    ) -> None:
        super().__init__()
        print("streaming mode is set to", cc.USE_BUFFER_CONV)

        if run is not None:
            config = os.path.join(run, "config.gin")

            # PARSE CONFIGURATION FILES
            gin.clear_config()
            gin.parse_config_file(config)

        # BUILD MODEL
        model = Prior()

        if run is not None:
            ckpts = pathlib.Path(run).rglob("*.ckpt")
            ckpts = map(str, ckpts)
            ckpts = sorted(ckpts, key=lambda x: "last" in x, reverse=True)
            ckpt = next(iter(ckpts))
            ckpt = torch.load(ckpt, map_location="cpu")["state_dict"]
            model.load_state_dict(ckpt)

        model.eval()

        self.encoder = model.encoder
        self.decoder = model.decoder

        # RETRIEVE INPUT/OUTPUT DETAILS
        self.seq_len = gin.get_bindings(
            "attention.MultiHeadAlibiAttention")["max_seq_len"]
        self.num_rave_quantizers = gin.get_bindings(
            "attention.MultivariateEmbedding")["num_quantizers"]
        self.num_tokens = gin.get_bindings(
            "attention.MultivariateEmbedding")["num_tokens"]
        self.has_encoder = self.encoder is not None

        num_inputs = self.num_rave_quantizers
        self.encoder_input_type = "none"
        self.encoder_num_tokens = 0
        self.feature_vae = None
        self.from_continuous = from_continuous

        if self.has_encoder:
            encoder_embedder = gin.get_bindings(
                "attention.Encoder")["embedder"]
            if isinstance(encoder_embedder(), Embedding):
                num_inputs += 1
                self.encoder_num_tokens = gin.get_bindings(
                    "encoder/attention.Embedding")["num_embeddings"]
                self.encoder_input_type = "discrete"
            elif isinstance(encoder_embedder(), FeatureEmbedding):
                if vae_path is not None:
                    self.encoder_input_type = "vae"
                    self.feature_vae = torch.jit.load(vae_path).eval()
                    num_inputs += self.feature_vae.latent_size
                else:
                    num_inputs += 163  # TODO: put that somewhere in configuration files
                    self.encoder_input_type = "full"
            else:
                raise ValueError(
                    f"Unknown encoder embedder {encoder_embedder}")

            self.encoder_ratio = gin.get_bindings(
                "decoder/attention.TransformerLayer")["encoder_out_ratio"]

        input_labels = [
            f"rave latent {i+1}" for i in range(self.num_rave_quantizers)
        ]

        if self.encoder_input_type == "none":
            pass
        elif self.encoder_input_type == "discrete":
            input_labels.append("semantic tokens")
        elif self.encoder_input_type == "vae":
            input_labels.extend([f"semantic latent {i}" for i in range(8)])
        elif self.encoder_input_type == "full":
            input_labels.extend([f"semantic feature {i}" for i in range(163)])
        else:
            raise ValueError(
                f"Encoder input type {self.encoder_input_type} not understood."
            )

        output_labels = [
            f"rave latent {i+1}" for i in range(self.num_rave_quantizers)
        ]
        output_labels.append("perplexity")

        self.register_buffer(
            "previous_step",
            torch.zeros(
                cc.MAX_BATCH_SIZE,
                1,
                self.num_rave_quantizers,
            ),
        )

        self.register_attribute("temperature", 1.)
        self.register_attribute("listen", initial_listen)
        self.register_attribute("reset", True)

        self.register_method(
            "forward",
            num_inputs,
            temporal_ratio,
            self.num_rave_quantizers + 1,
            temporal_ratio,
            input_labels=input_labels,
            output_labels=output_labels,
        )

        if self.has_encoder:
            # FORCE NN~ RATE DETECTION TO USE LARGER BUFFERS
            self.register_method(
                "dummy_method",
                1,
                self.encoder_ratio * temporal_ratio,
                1,
                self.encoder_ratio * temporal_ratio,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        input_rave = x[:, :self.num_rave_quantizers]

        if self.from_continuous:
            input_rave = self.quantize(input_rave)

        input_rave = torch.clamp(
            input_rave,
            min=0,
            max=self.num_tokens,
        ).long()
        input_rave = input_rave.permute(0, 2, 1)

        if self.reset[0]:
            for n, b in self.named_buffers():
                if "_cache_length" in n or "_relative_index" in n or "_state" in n:
                    b.zero_()
            self.set_reset(False)

        if self.has_encoder:
            assert hasattr(self, "encoder_ratio")
            if self.encoder_input_type == "discrete":
                semantic_tokens = x[:, self.num_rave_quantizers:]
                semantic_tokens = semantic_tokens.reshape(
                    batch_size,
                    semantic_tokens.shape[1],
                    -1,
                    self.encoder_ratio,
                )[..., -1]
                semantic_tokens = torch.clamp(
                    semantic_tokens,
                    0,
                    self.encoder_num_tokens,
                ).long()[:, 0]
                encoder_out = self.encoder(semantic_tokens)
            elif self.encoder_input_type == "vae":
                assert self.feature_vae is not None
                latents = x[:, self.num_rave_quantizers:]
                latents = latents.reshape(
                    batch_size,
                    latents.shape[1],
                    -1,
                    self.encoder_ratio,
                )[..., -1]
                features = self.feature_vae(latents).permute(0, 2, 1)
                encoder_out = self.encoder(features)
            elif self.encoder_input_type == "full":
                features = x[:, self.num_rave_quantizers:]
                features = features.reshape(
                    batch_size,
                    features.shape[1],
                    -1,
                    self.encoder_ratio,
                )[..., -1]
                encoder_out = self.encoder(features.permute(0, 2, 1))
            else:
                raise ValueError(
                    f"encoder input type {self.encoder_input_type} not understood"
                )
        else:
            encoder_out = None

        if self.listen[0]:
            _, perp = self.decoder.sample(
                input_rave,
                encoder_out,
                self.temperature[0],
            )
            self.previous_step[:batch_size].copy_(input_rave[:, -1:])

            if self.from_continuous:
                input_rave = self.dequantize(input_rave)

            out = torch.cat([input_rave, perp[..., None]], -1)
            return out.permute(0, 2, 1)
        else:
            sample_list = []
            perp_list = []

            if encoder_out is not None:
                encoder_out = encoder_out.repeat_interleave(
                    self.encoder_ratio, 1)

            for t in range(input_rave.shape[1]):
                if encoder_out is not None:
                    current_encoder_out = encoder_out[:, t:t + 1]
                else:
                    current_encoder_out = None

                sample, perp = self.decoder.sample(
                    self.previous_step[:batch_size],
                    current_encoder_out,
                    self.temperature[0],
                )

                self.previous_step[:batch_size].copy_(sample)
                sample_list.append(sample)
                perp_list.append(perp)

            samples = torch.cat(sample_list, 1)
            perps = torch.stack(perp_list, 1)

            if self.from_continuous:
                samples = self.dequantize(samples)

            out = torch.cat([samples, perps], -1).permute(0, 2, 1)
            return out

    def dummy_method(self, x: torch.Tensor) -> torch.Tensor:
        # FORCE MINIMUM BUFFER SIZE TO SEMANTIC RATE
        return x

    def quantize(self, x):
        x = x / 2
        x = .5 * (1 + torch.erf(x / math.sqrt(2)))
        x = torch.floor(x * self.num_tokens - 1).long()
        return x

    def dequantize(self, x):
        x = x.float()
        x = (x + torch.rand_like(x)) / self.num_tokens
        x = torch.erfinv(2 * x - 1) * math.sqrt(2) * 2
        return x

    @torch.jit.export
    def get_temperature(self) -> float:
        return self.temperature[0]

    @torch.jit.export
    def set_temperature(self, temperature: float) -> int:
        if temperature < 0:
            return -1
        self.temperature = (temperature, )
        return 0

    @torch.jit.export
    def get_listen(self) -> bool:
        return self.listen[0]

    @torch.jit.export
    def set_listen(self, listen: bool) -> int:
        self.listen = (listen, )
        return 0

    @torch.jit.export
    def get_reset(self) -> bool:
        return self.reset[0]

    @torch.jit.export
    def set_reset(self, reset: bool) -> int:
        self.reset = (reset, )
        return 0
