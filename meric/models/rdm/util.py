"""Small diffusion-model building blocks used by the Meric Music Head.

Adapted from OpenAI improved-diffusion and guided-diffusion under the MIT
License.
"""

import math

import torch


def timestep_embedding(timesteps, dim, max_period=10_000, repeat_only=False):
    """Create sinusoidal timestep embeddings for a one-dimensional batch."""
    if repeat_only:
        return timesteps[:, None].repeat(1, dim)

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """Zero all parameters in a module and return it."""
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
