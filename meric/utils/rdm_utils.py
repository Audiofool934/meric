#!/usr/bin/env python3
"""
Shared utilities for RDM training and inference.

Contains:
  - EMAModel: Exponential Moving Average for model parameters
  - WarmupCosineScheduler: Linear warmup + cosine decay LR schedule
  - Min-SNR loss weighting functions
  - load_rdm_checkpoint: Unified checkpoint loader with auto-detection
  - batch_ddim_sample: Canonical batch DDIM sampling for Stage 1 inference
"""

import hashlib
import math
import os
from numbers import Integral
from pathlib import Path

import numpy as np
import torch


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of all parameters, updated each step as:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMAModel(model, decay=0.9999)
        # After each optimizer step:
        ema.update(model)
        # For validation/saving:
        ema.apply_shadow(model)
        val_loss = validate(model, ...)
        ema.restore(model)
    """

    def __init__(self, model, decay=0.9999):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = decay
        # Get the underlying model if wrapped in DataParallel
        base = model.module if hasattr(model, "module") else model
        self.shadow = {name: param.data.clone() for name, param in base.named_parameters()}
        self.backup = {}

    def update(self, model):
        """Update shadow parameters with current model parameters."""
        base = model.module if hasattr(model, "module") else model
        for name, param in base.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Replace model parameters with shadow (EMA) parameters. Call restore() after."""
        base = model.module if hasattr(model, "module") else model
        self.backup = {name: param.data.clone() for name, param in base.named_parameters()}
        for name, param in base.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore model parameters from backup (undo apply_shadow)."""
        base = model.module if hasattr(model, "module") else model
        for name, param in base.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict):
        expected = set(self.shadow)
        received = set(state_dict)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            raise RuntimeError(f"EMA state mismatch: missing={missing}, unexpected={unexpected}")
        for name, tensor in state_dict.items():
            if tensor.shape != self.shadow[name].shape:
                raise RuntimeError(
                    f"EMA tensor {name!r} has shape {tuple(tensor.shape)}, expected {tuple(self.shadow[name].shape)}"
                )
            self.shadow[name].copy_(tensor)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        if total_epochs < 1:
            raise ValueError("total_epochs must be at least 1")
        if not 0 <= warmup_epochs <= total_epochs:
            raise ValueError("warmup_epochs must be between 0 and total_epochs")
        if not 0 <= min_lr <= base_lr:
            raise ValueError("min_lr must be between 0 and base_lr")
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        self._set_lr(self.get_lr())

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        self._set_lr(lr)

    def get_lr(self):
        epoch = min(self.current_epoch, self.total_epochs - 1)
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs

        decay_span = max(self.total_epochs - self.warmup_epochs - 1, 1)
        progress = min(max(epoch - self.warmup_epochs, 0) / decay_span, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        """Return enough state to resume the exact learning-rate schedule."""
        return {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict):
        """Restore state and reject a schedule configured differently."""
        expected = {
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }
        for key, value in expected.items():
            if key not in state_dict:
                raise ValueError(f"Learning-rate scheduler state is missing {key!r}")
            if state_dict[key] != value:
                raise ValueError(
                    f"Learning-rate scheduler mismatch for {key}: checkpoint={state_dict[key]!r}, configured={value!r}"
                )
        current_epoch = state_dict.get("current_epoch")
        if not isinstance(current_epoch, int) or not 0 <= current_epoch <= self.total_epochs:
            raise ValueError(f"Invalid scheduler current_epoch: {current_epoch!r}")
        self.current_epoch = current_epoch
        self._set_lr(self.get_lr())


def normalize_gradients(parameters, sample_count):
    """Turn summed per-sample gradients into a mean before clipping and stepping."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    scale = 1.0 / sample_count
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def capture_rng_state():
    """Capture PyTorch, CUDA, and NumPy RNG state in a weights-only-safe form."""
    bit_generator, numpy_state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_bit_generator": bit_generator,
        # PyTorch 2.5 cannot serialize uint32 typed storage. MT19937 values
        # fit exactly in int64 and are converted back to uint32 on restore.
        "numpy_state": torch.from_numpy(numpy_state.astype(np.int64)),
        "numpy_position": position,
        "numpy_has_gauss": has_gauss,
        "numpy_cached_gaussian": cached_gaussian,
    }


def restore_rng_state(state):
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        return False

    required = {
        "torch",
        "cuda",
        "numpy_bit_generator",
        "numpy_state",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RNG state is missing fields: {missing}")

    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state["cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG state count does not match visible devices: "
                f"checkpoint={len(cuda_states)}, visible={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in cuda_states])

    numpy_state = state["numpy_state"]
    if not isinstance(numpy_state, torch.Tensor):
        raise TypeError("numpy_state must be a torch.Tensor")
    np.random.set_state(
        (
            state["numpy_bit_generator"],
            numpy_state.cpu().numpy().astype(np.uint32, copy=False),
            state["numpy_position"],
            state["numpy_has_gauss"],
            state["numpy_cached_gaussian"],
        )
    )
    return True


def atomic_torch_save(value, path):
    """Write a PyTorch checkpoint atomically within its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of a regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_snr(noise_scheduler, timesteps):
    """Compute signal-to-noise ratio for given timesteps.

    SNR(t) = alpha_cumprod(t) / (1 - alpha_cumprod(t))
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_t = alphas_cumprod[timesteps]
    snr = alpha_t / (1.0 - alpha_t)
    return snr


def compute_snr_weights(snr, gamma=5.0):
    """Compute Min-SNR-gamma loss weights.

    weight(t) = min(SNR(t), gamma) / SNR(t)
    This down-weights high-SNR (easy) timesteps, balancing gradients.
    """
    return torch.clamp(snr, max=gamma) / snr


def compute_v_target(noise_scheduler, x0, noise, timesteps):
    """Compute v-prediction target: v = alpha_t * noise - sigma_t * x0.

    Args:
        noise_scheduler: DDPMScheduler with alphas_cumprod
        x0: clean signal [B, D]
        noise: sampled noise [B, D]
        timesteps: [B] timestep indices

    Returns:
        v_target: [B, D]
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_t = alphas_cumprod[timesteps].sqrt()
    sigma_t = (1.0 - alphas_cumprod[timesteps]).sqrt()

    # Expand for broadcasting: [B] -> [B, 1]
    alpha_t = alpha_t.unsqueeze(-1)
    sigma_t = sigma_t.unsqueeze(-1)

    return alpha_t * noise - sigma_t * x0


def load_rdm_checkpoint(ckpt_path, context_dim, device, *, checkpoint=None):
    """Load RDM checkpoint with auto-detection of scheduler config.

    New checkpoints store scheduler_config (beta_schedule, prediction_type, etc.).
    Old checkpoints fall back to linear + sample defaults.

    Args:
        ckpt_path: Path to checkpoint .pth file
        context_dim: Context embedding dimension (1024 for CLIP, 2048 for Qwen3-VL)
        device: torch device

    Returns:
        (model, scheduler, scale_factor)
    """
    from diffusers import DDIMScheduler

    from meric.models.rdm.latentmlp import SimpleMLP

    if checkpoint is None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

    # Auto-detect model width from checkpoint
    state_dict = checkpoint["model_state_dict"]
    model_channels = state_dict["input_proj.weight"].shape[0]

    # Auto-detect depth from checkpoint (prefer scheduler_config, fall back to inspecting state_dict keys)
    sched_config = checkpoint.get("scheduler_config", {})
    num_res_blocks = sched_config.get("num_res_blocks")
    if num_res_blocks is None:
        block_idx = set()
        for k in state_dict.keys():
            if k.startswith("res_blocks."):
                try:
                    block_idx.add(int(k.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        num_res_blocks = max(block_idx) + 1 if block_idx else 12

    model = SimpleMLP(
        in_channels=512,
        time_embed_dim=256,
        model_channels=model_channels,
        bottleneck_channels=model_channels,
        out_channels=512,
        num_res_blocks=num_res_blocks,
        dropout=0.0,  # inference mode, dropout irrelevant
        use_context=True,
        context_channels=context_dim,
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Auto-detect scheduler config from checkpoint (new format)
    # Fall back to old defaults for backward compatibility
    sched_config = checkpoint.get("scheduler_config", {})
    beta_schedule = sched_config.get("beta_schedule", "linear")
    prediction_type = sched_config.get("prediction_type", "sample")
    beta_start = sched_config.get("beta_start", 0.0015)
    beta_end = sched_config.get("beta_end", 0.0195)
    scale_factor = sched_config.get("scale_factor", 10.0)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type=prediction_type,
    )

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"  RDM loaded (epoch={epoch}, val_loss={val_loss}, context_dim={context_dim})")
    print(f"  Scheduler: {beta_schedule}, prediction: {prediction_type}, scale: {scale_factor}")

    return model, scheduler, scale_factor


@torch.no_grad()
def batch_ddim_sample(
    model,
    conditions,
    scheduler,
    device,
    num_steps=20,
    guidance_scale=2.0,
    scale_factor=10.0,
    output_dim=512,
    seed=42,
    batch_size=512,
    model_kwargs=None,
):
    """Batch DDIM sampling for Stage 1 RDM inference.

    Processes all conditions in chunks through a shared timestep loop.
    ~100x faster than per-sample inference for large datasets.

    Args:
        model: RDM denoiser (SimpleMLP)
        conditions: np.ndarray or torch.Tensor [N, cond_dim]
        scheduler: DDIMScheduler (used directly) or DDPMScheduler (DDIM created from config)
        device: torch device
        num_steps: DDIM denoising steps (default 20)
        guidance_scale: classifier-free guidance scale (1.0 = no guidance)
        scale_factor: output scaling divisor (default 10.0)
        output_dim: output embedding dimension (512 for MuQ, 2048 for Qwen3-VL)
        seed: random seed for reproducible initial noise
        batch_size: max samples per GPU batch (512 is fine for MLP)
        model_kwargs: extra kwargs passed to model forward (e.g., direction='reverse')

    Returns:
        np.ndarray [N, output_dim] predicted embeddings
    """
    from diffusers import DDIMScheduler as _DDIMScheduler

    # Create DDIM scheduler from config (handles both DDIM and DDPM inputs)
    ddim = _DDIMScheduler(
        num_train_timesteps=scheduler.config.num_train_timesteps,
        beta_start=scheduler.config.beta_start,
        beta_end=scheduler.config.beta_end,
        beta_schedule=scheduler.config.beta_schedule,
        clip_sample=False,
        prediction_type=scheduler.config.prediction_type,
    )
    ddim.set_timesteps(num_steps)

    # Convert conditions to numpy if tensor
    if isinstance(conditions, torch.Tensor):
        conditions = conditions.cpu().numpy()
    conditions = np.asarray(conditions)
    if conditions.ndim != 2 or conditions.shape[0] < 1:
        raise ValueError(f"conditions must have shape [N, D] with N >= 1, got {conditions.shape}")
    if not np.issubdtype(conditions.dtype, np.number) or not np.isfinite(conditions).all():
        raise ValueError("conditions must be a finite numeric array")
    for value, name in ((num_steps, "num_steps"), (batch_size, "batch_size"), (output_dim, "output_dim")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer between 0 and 2^63 - 1")
    seed = int(seed)
    if model_kwargs is not None and not isinstance(model_kwargs, dict):
        raise TypeError("model_kwargs must be a dictionary or None")

    N = conditions.shape[0]
    model_kwargs = model_kwargs or {}
    all_preds = []
    generator = torch.Generator(device=device).manual_seed(seed)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        cond = torch.tensor(conditions[start:end], dtype=torch.float32, device=device)
        uncond = torch.zeros_like(cond)

        x = torch.randn(B, output_dim, device=device, generator=generator) * ddim.init_noise_sigma

        for t in ddim.timesteps:
            t_batch = t.expand(B).to(device)
            pred_cond = model(x=x, timesteps=t_batch, context=cond, **model_kwargs)

            if guidance_scale != 1.0:
                pred_uncond = model(x=x, timesteps=t_batch, context=uncond, **model_kwargs)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            x = ddim.step(pred, t, x).prev_sample

        all_preds.append((x / scale_factor).cpu().numpy())

    return np.concatenate(all_preds, axis=0)
