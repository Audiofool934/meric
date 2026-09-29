"""Residual MLP denoiser used by the Meric Music Head.

The timestep embedding and zero-initialized residual/output projections are
adapted from OpenAI improved-diffusion and CompVis latent-diffusion under their
respective permissive licenses.
"""

import math
from numbers import Integral

import torch
import torch.nn as nn

from .util import timestep_embedding, zero_module


def _as_batch_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    """Return a [batch, features] view without dropping a batch of size one."""
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim > 2:
        value = value.flatten(start_dim=1)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a feature matrix, got shape {tuple(value.shape)}")
    return value


class ResBlock(nn.Module):
    """Timestep-conditioned residual MLP block with optional context."""

    def __init__(self, channels, mid_channels, emb_channels, dropout, use_context=False, context_channels=512):
        super().__init__()
        self.use_context = use_context
        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )
        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(mid_channels, channels, bias=True)),
        )
        if use_context:
            self.context_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, mid_channels, bias=True),
            )

    def forward(self, x, emb, context=None):
        h = self.in_layers(x) + self.emb_layers(emb)
        if self.use_context:
            if context is None:
                raise ValueError("A context embedding is required by this Music Head")
            h = h + self.context_layers(context)
        return x + self.out_layers(h)


class SimpleMLP(nn.Module):
    """Residual denoiser mapping noisy MuQ vectors and context to MuQ vectors."""

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        bottleneck_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        use_context=False,
        context_channels=512,
    ):
        super().__init__()
        dimensions = {
            "in_channels": in_channels,
            "time_embed_dim": time_embed_dim,
            "model_channels": model_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_res_blocks": num_res_blocks,
        }
        invalid = {
            name: value
            for name, value in dimensions.items()
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        }
        if invalid:
            raise ValueError(f"Expected positive integer Music Head dimensions, got {invalid}")
        if model_channels < 2:
            raise ValueError("model_channels must be at least 2 for timestep embeddings")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if use_context and (
            isinstance(context_channels, bool) or not isinstance(context_channels, Integral) or context_channels < 1
        ):
            raise ValueError("context_channels must be a positive integer when context is enabled")

        self.model_channels = model_channels
        self.use_context = use_context
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    model_channels,
                    bottleneck_channels,
                    time_embed_dim,
                    dropout,
                    use_context=use_context,
                    context_channels=context_channels,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def forward(self, x, timesteps=None, context=None, **_kwargs):
        """Denoise a batch of feature vectors."""
        x = _as_batch_matrix(x, "x")
        if timesteps is None:
            raise ValueError("timesteps are required")
        timesteps = timesteps.to(x.device).reshape(-1)
        if timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps and x must have the same batch size")
        if x.shape[1] != self.input_proj.in_features:
            raise ValueError(f"x must have {self.input_proj.in_features} features, got {x.shape[1]}")

        if context is not None:
            context = _as_batch_matrix(context, "context")
            if context.shape[0] != x.shape[0]:
                raise ValueError("context and x must have the same batch size")
            expected_context = self.res_blocks[0].context_layers[1].in_features if self.use_context else None
            if expected_context is not None and context.shape[1] != expected_context:
                raise ValueError(f"context must have {expected_context} features, got {context.shape[1]}")
        elif self.use_context:
            raise ValueError("context is required for this Music Head")

        x = self.input_proj(x)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        for block in self.res_blocks:
            x = block(x, emb, context)
        return self.out(x)
