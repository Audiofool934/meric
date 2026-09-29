#!/usr/bin/env python3
"""Fine-tune the Meric Music Head on paired visual, text, and MuQ features.

The camera-ready recipe inherits its noise scheduler from pretraining, applies
EMA and min-SNR weighting, and mixes ARIA pairs with AudioSet-Music replay. See
``docs/REPRODUCIBILITY.md`` for the canonical invocation and data requirements.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from meric.data.unified_dataset import ReplayWrapper, SFTMixedDataset, UnifiedEmbeddingDataset  # noqa: E402
from meric.models.rdm.latentmlp import SimpleMLP  # noqa: E402
from meric.utils.rdm_utils import (  # noqa: E402
    EMAModel,
    WarmupCosineScheduler,
    atomic_torch_save,
    capture_rng_state,
    compute_snr,
    compute_snr_weights,
    compute_v_target,
    normalize_gradients,
    restore_rng_state,
    sha256_file,
)


def get_prediction_target(noise_scheduler, audio_feat, noise, timesteps):
    """Get the prediction target based on scheduler's prediction_type."""
    ptype = noise_scheduler.config.prediction_type
    if ptype == "epsilon":
        return noise
    elif ptype == "sample":
        return audio_feat
    elif ptype == "v_prediction":
        return compute_v_target(noise_scheduler, audio_feat, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {ptype}")


def train_epoch(
    model,
    train_loader,
    replay_loader,
    optimizer,
    noise_scheduler,
    device,
    scale_factor=10.0,
    unconditional_prob=0.1,
    max_grad_norm=1.0,
    grad_accum=1,
    replay_ratio=0.15,
    snr_gamma=0.0,
    ema=None,
):
    """Train one epoch and return sample-weighted aggregate and per-type losses."""
    model.train()
    total_loss = 0.0
    image_loss_sum = 0.0
    text_loss_sum = 0.0
    replay_loss_sum = 0.0
    total_count = 0
    image_count = 0
    text_count = 0
    replay_count = 0
    n_steps = 0

    replay_iter = iter(replay_loader) if replay_loader else None

    optimizer.zero_grad()
    accum_count = 0
    accum_samples = 0

    for _batch_idx, (cond_batch, muq_batch, type_batch) in enumerate(tqdm(train_loader, desc="Training")):
        # Optionally mix in replay samples
        if replay_iter is not None and replay_ratio > 0:
            try:
                r_cond, r_muq, r_type = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                r_cond, r_muq, r_type = next(replay_iter)

            n_replay = max(1, int(cond_batch.shape[0] * replay_ratio))
            n_replay = min(n_replay, r_cond.shape[0])
            cond_batch = torch.cat([cond_batch[:-n_replay], r_cond[:n_replay]])
            muq_batch = torch.cat([muq_batch[:-n_replay], r_muq[:n_replay]])
            type_batch = torch.cat([type_batch[:-n_replay], r_type[:n_replay]])

        cond_batch = cond_batch.to(device)
        muq_batch = muq_batch.to(device)
        type_batch = type_batch.to(device)
        batch_size = cond_batch.shape[0]

        audio_feat = muq_batch * scale_factor
        noise = torch.randn_like(audio_feat)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_audio_feat = noise_scheduler.add_noise(audio_feat, noise, timesteps)

        # CFG dropout
        context = cond_batch.clone()
        if unconditional_prob > 0:
            mask = torch.rand(batch_size, 1, device=device) < unconditional_prob
            context = context * (~mask).float()

        target = get_prediction_target(noise_scheduler, audio_feat, noise, timesteps)

        pred = model(x=noisy_audio_feat, timesteps=timesteps, context=context)

        per_sample_loss = nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)

        # Min-SNR weighting
        if snr_gamma > 0:
            snr = compute_snr(noise_scheduler, timesteps)
            snr_weights = compute_snr_weights(snr, gamma=snr_gamma)
            per_sample_loss = per_sample_loss * snr_weights

        if not torch.isfinite(per_sample_loss).all():
            raise FloatingPointError("Training loss became non-finite")

        # Track the optimized, scheduler-weighted loss by condition type.
        is_image = type_batch == 0
        is_text = type_batch == 1
        is_replay = type_batch == 2
        if is_image.any():
            image_loss_sum += per_sample_loss[is_image].sum().item()
            image_count += is_image.sum().item()
        if is_text.any():
            text_loss_sum += per_sample_loss[is_text].sum().item()
            text_count += is_text.sum().item()
        if is_replay.any():
            replay_loss_sum += per_sample_loss[is_replay].sum().item()
            replay_count += is_replay.sum().item()

        per_sample_loss.sum().backward()
        accum_count += 1
        accum_samples += batch_size

        if accum_count >= grad_accum:
            normalize_gradients(model.parameters(), accum_samples)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            accum_samples = 0
            n_steps += 1

            if ema is not None:
                ema.update(model)

        total_loss += per_sample_loss.sum().item()
        total_count += batch_size

    # Final accumulated gradients
    if accum_count > 0:
        normalize_gradients(model.parameters(), accum_samples)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1
        if ema is not None:
            ema.update(model)

    if total_count < 1:
        raise RuntimeError("Training dataloader produced no samples")
    return {
        "loss": total_loss / total_count,
        "image_loss": image_loss_sum / max(image_count, 1),
        "text_loss": text_loss_sum / max(text_count, 1),
        "replay_loss": replay_loss_sum / max(replay_count, 1),
        "image_count": image_count,
        "text_count": text_count,
        "replay_count": replay_count,
        "optimizer_steps": n_steps,
    }


def validate(model, val_loader, noise_scheduler, device, scale_factor=10.0, snr_gamma=0.0):
    """Validate with separate image/text loss tracking."""
    model.eval()
    total_loss = 0.0
    image_loss_sum = 0.0
    text_loss_sum = 0.0
    total_count = 0
    image_count = 0
    text_count = 0

    with torch.no_grad():
        for cond_batch, muq_batch, type_batch in val_loader:
            cond_batch = cond_batch.to(device)
            muq_batch = muq_batch.to(device)
            type_batch = type_batch.to(device)
            batch_size = cond_batch.shape[0]

            audio_feat = muq_batch * scale_factor
            noise = torch.randn_like(audio_feat)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            noisy_audio_feat = noise_scheduler.add_noise(audio_feat, noise, timesteps)

            target = get_prediction_target(noise_scheduler, audio_feat, noise, timesteps)

            pred = model(x=noisy_audio_feat, timesteps=timesteps, context=cond_batch)

            per_sample_loss = nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)

            if snr_gamma > 0:
                snr = compute_snr(noise_scheduler, timesteps)
                snr_weights = compute_snr_weights(snr, gamma=snr_gamma)
                per_sample_loss = per_sample_loss * snr_weights

            if not torch.isfinite(per_sample_loss).all():
                raise FloatingPointError("Validation loss became non-finite")

            is_image = type_batch == 0
            is_text = type_batch == 1
            if is_image.any():
                image_loss_sum += per_sample_loss[is_image].sum().item()
                image_count += is_image.sum().item()
            if is_text.any():
                text_loss_sum += per_sample_loss[is_text].sum().item()
                text_count += is_text.sum().item()

            total_loss += per_sample_loss.sum().item()
            total_count += batch_size

    if total_count < 1:
        raise RuntimeError("Validation dataloader produced no samples")
    return {
        "loss": total_loss / total_count,
        "image_loss": image_loss_sum / max(image_count, 1),
        "text_loss": text_loss_sum / max(text_count, 1),
        "image_count": image_count,
        "text_count": text_count,
    }


def resolve_scheduler_config(args, checkpoint=None):
    """Resolve noise scheduler config: CLI override > checkpoint > defaults.

    Priority:
      1. Explicit CLI args (--beta-schedule / --prediction-type != "auto")
      2. scheduler_config stored in pretrained checkpoint
      3. Fallback defaults (linear + sample for older checkpoints without config)
    """
    # Backward-compatible defaults for checkpoints without scheduler metadata.
    beta_schedule = "linear"
    prediction_type = "sample"

    # Read from checkpoint if available
    if checkpoint is not None:
        sched_config = checkpoint.get("scheduler_config", {})
        if sched_config:
            beta_schedule = sched_config.get("beta_schedule", beta_schedule)
            prediction_type = sched_config.get("prediction_type", prediction_type)

    # CLI overrides (only if not "auto")
    if args.beta_schedule != "auto":
        beta_schedule = args.beta_schedule
    if args.prediction_type != "auto":
        prediction_type = args.prediction_type

    return beta_schedule, prediction_type


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Meric Music Head")
    # Data
    parser.add_argument(
        "--manifest", type=str, default="data/unified/meta/sft_train.jsonl", help="SFT training JSONL manifest"
    )
    parser.add_argument(
        "--val-manifest",
        "--val_manifest",
        type=str,
        default="data/unified/meta/test_sft.jsonl",
        help="SFT validation JSONL manifest",
    )
    parser.add_argument(
        "--replay-manifest",
        "--replay_manifest",
        type=str,
        default=None,
        help="Replay buffer JSONL manifest (e.g., visual_train.jsonl)",
    )
    parser.add_argument(
        "--replay-filter",
        "--replay_filter",
        type=str,
        default="audioset_music",
        help="Filter replay manifest to this dataset (default: audioset_music)",
    )
    parser.add_argument(
        "--replay-ratio",
        "--replay_ratio",
        type=float,
        default=0.15,
        help="Fraction of batch replaced by replay samples (default: 0.15)",
    )
    parser.add_argument(
        "--replay-max",
        "--replay_max",
        type=int,
        default=50000,
        help="Max replay samples to load (default: 50000, for memory)",
    )
    parser.add_argument(
        "--data-root", "--data_root", type=str, default=".", help="Root dir for resolving relative paths in manifest"
    )
    parser.add_argument("--output-dir", "--output_dir", type=str, default="outputs/checkpoints/rdm_sft_v3")
    parser.add_argument("--pretrained", type=str, default="outputs/checkpoints/rdm_pretrain/best_model.pth")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (e.g. outputs/checkpoints/rdm_sft_v3/checkpoint_epoch0020.pth)",
    )

    # Model
    parser.add_argument("--context-dim", "--context_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate in ResBlocks (default: 0.05)")

    # Noise schedule (auto = inherit from pretrained checkpoint)
    parser.add_argument(
        "--beta-schedule",
        "--beta_schedule",
        type=str,
        default="auto",
        choices=["auto", "linear", "squaredcos_cap_v2", "scaled_linear"],
        help="Noise schedule (default: auto = read from checkpoint)",
    )
    parser.add_argument(
        "--prediction-type",
        "--prediction_type",
        type=str,
        default="auto",
        choices=["auto", "sample", "epsilon", "v_prediction"],
        help="Prediction target (default: auto = read from checkpoint)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256)
    parser.add_argument(
        "--grad-accum",
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch size * accumulation steps)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", type=int, default=5)
    parser.add_argument("--min-lr", "--min_lr", type=float, default=1e-7, help="Minimum LR for cosine decay")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scale-factor", "--scale_factor", type=float, default=10.0)
    parser.add_argument("--unconditional-prob", "--unconditional_prob", type=float, default=0.1)
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=4,
        help="Worker processes per dataloader (default: 4)",
    )
    parser.add_argument(
        "--preload-workers",
        "--preload_workers",
        type=int,
        default=32,
        help="Threads used to validate and preload cached embeddings (default: 32)",
    )

    # EMA
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")

    # Min-SNR
    parser.add_argument(
        "--snr-gamma", "--snr_gamma", type=float, default=5.0, help="Min-SNR gamma for loss weighting (0 to disable)"
    )

    # Checkpointing
    parser.add_argument("--save-every", "--save_every", type=int, default=10)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-only",
        "--image_only",
        action="store_true",
        help="Train with image pairs only (skip text-to-MuQ pairs)",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be at least 1")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.num_workers < 0 or args.preload_workers < 1:
        parser.error("--num-workers must be non-negative and --preload-workers must be at least 1")
    if args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        parser.error("require --lr > 0 and 0 <= --min-lr <= --lr")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if not 0 <= args.replay_ratio <= 1:
        parser.error("--replay-ratio must be between 0 and 1")
    if not 0 <= args.unconditional_prob <= 1:
        parser.error("--unconditional-prob must be between 0 and 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be at least 0 and less than 1")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be at least 0 and less than 1")
    if args.scale_factor <= 0 or args.weight_decay < 0 or args.max_grad_norm < 0 or args.snr_gamma < 0:
        parser.error("scale must be positive; decay, clipping, and SNR values must be non-negative")
    if args.context_dim < 1 or args.replay_max < 1:
        parser.error("--context-dim and --replay-max must be at least 1")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32 - 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        device_ids = [int(value) for value in args.devices.split(",") if value.strip()]
    except ValueError:
        parser.error("--devices must be a comma-separated list of CUDA indices")
    if not device_ids or len(device_ids) != len(set(device_ids)) or any(value < 0 for value in device_ids):
        parser.error("--devices must contain one or more distinct, non-negative CUDA indices")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Music Head training")
    if any(value >= torch.cuda.device_count() for value in device_ids):
        parser.error(f"--devices references an unavailable CUDA index; {torch.cuda.device_count()} device(s) visible")
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{device_ids[0]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint metadata before constructing the scheduler or reporting the run.
    pretrained_ckpt = None
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
    elif args.pretrained:
        pretrained_ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True, mmap=True)

    beta_schedule, prediction_type = resolve_scheduler_config(args, resume_ckpt or pretrained_ckpt)

    eff_batch = args.batch_size * args.grad_accum
    print("=" * 70)
    print("Meric Music Head Fine-tuning")
    print("=" * 70)
    print(f"Manifest:         {args.manifest}")
    print(f"Val manifest:     {args.val_manifest}")
    print(f"Replay manifest:  {args.replay_manifest}")
    print(f"Replay filter:    {args.replay_filter}")
    print(f"Pretrained:       {args.pretrained}")
    print(f"Resume:           {args.resume}")
    print(f"Replay ratio:     {args.replay_ratio}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size} x {args.grad_accum} accum = {eff_batch} effective")
    print(f"Learning rate:    {args.lr} (warmup {args.warmup_epochs} epochs -> cosine to {args.min_lr})")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Grad clip:        {args.max_grad_norm}")
    print(f"Dropout:          {args.dropout}")
    print(f"EMA decay:        {args.ema_decay}")
    print(f"Noise schedule:   {beta_schedule}")
    print(f"Prediction type:  {prediction_type}")
    print(f"Min-SNR gamma:    {args.snr_gamma}")
    print(f"Devices:          {device_ids}")
    print(f"Image only:       {args.image_only}")
    print(f"Data workers:     {args.num_workers} process(es), {args.preload_workers} preload thread(s)")

    # Datasets
    print("\nLoading train dataset...")
    train_dataset = SFTMixedDataset(
        args.manifest,
        preload=True,
        data_root=args.data_root,
        image_only=args.image_only,
        preload_workers=args.preload_workers,
    )

    print("\nLoading val dataset...")
    val_dataset = SFTMixedDataset(
        args.val_manifest,
        preload=True,
        data_root=args.data_root,
        image_only=args.image_only,
        preload_workers=args.preload_workers,
    )

    # Replay buffer
    replay_loader = None
    replay_dataset = None
    if args.replay_manifest:
        print("\nLoading replay buffer...")
        replay_filter = [args.replay_filter] if args.replay_filter else None
        replay_base = UnifiedEmbeddingDataset(
            args.replay_manifest,
            cond_key="qwen3vl",
            target_key="muq",
            preload=True,
            data_root=args.data_root,
            filter_datasets=replay_filter,
            max_samples=args.replay_max,
            preload_workers=args.preload_workers,
        )
        replay_dataset = ReplayWrapper(replay_base)
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        print(f"  Replay buffer: {len(replay_dataset)} samples (ratio={args.replay_ratio})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("\nDataloader:")
    print(f"  Train: {len(train_dataset)} pairs/epoch")
    print(f"  Val:   {len(val_dataset)} pairs/epoch")

    # Model
    model = SimpleMLP(
        in_channels=512,
        time_embed_dim=256,
        model_channels=1536,
        bottleneck_channels=1536,
        out_channels=512,
        num_res_blocks=12,
        dropout=args.dropout,
        use_context=True,
        context_channels=args.context_dim,
    )

    # Load weights
    start_epoch = 0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_image_loss": [],
        "train_text_loss": [],
        "train_replay_loss": [],
        "val_image_loss": [],
        "val_text_loss": [],
        "lr": [],
    }

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = resume_ckpt
        checkpoint_scheduler = ckpt.get("scheduler_config", {})
        expected_scheduler = {
            "beta_schedule": beta_schedule,
            "prediction_type": prediction_type,
            "scale_factor": args.scale_factor,
            "context_dim": args.context_dim,
        }
        mismatches = {
            key: (checkpoint_scheduler[key], value)
            for key, value in expected_scheduler.items()
            if key in checkpoint_scheduler and checkpoint_scheduler[key] != value
        }
        if mismatches:
            parser.error(f"resume checkpoint configuration does not match this run: {mismatches}")

        previous_config = ckpt.get("training_config", {})
        exact_resume_fields = (
            "batch_size",
            "grad_accum",
            "lr",
            "warmup_epochs",
            "min_lr",
            "weight_decay",
            "scale_factor",
            "unconditional_prob",
            "dropout",
            "ema_decay",
            "snr_gamma",
            "replay_ratio",
            "replay_max",
            "replay_filter",
            "context_dim",
            "image_only",
            "devices",
            "seed",
        )
        config_mismatches = {
            key: (previous_config[key], getattr(args, key))
            for key in exact_resume_fields
            if key in previous_config and previous_config[key] != getattr(args, key)
        }
        if config_mismatches:
            parser.error(f"resume training arguments do not match the checkpoint: {config_mismatches}")

        # Prefer raw weights for continued training (EMA weights are for inference)
        raw_state = ckpt.get("raw_model_state_dict", ckpt["model_state_dict"])
        model.load_state_dict(raw_state)
        start_epoch = ckpt["epoch"]
        if not isinstance(start_epoch, int) or start_epoch < 0:
            parser.error(f"resume checkpoint has an invalid epoch: {start_epoch!r}")
        if start_epoch >= args.epochs:
            parser.error(f"checkpoint epoch {start_epoch} is not less than --epochs {args.epochs}")
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        hist_path = output_dir / "training_history.json"
        if "history" in ckpt:
            history = ckpt["history"]
        elif hist_path.is_file():
            with hist_path.open(encoding="utf-8") as f:
                history = json.load(f)
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        print(f"  Scheduler: {beta_schedule}, prediction: {prediction_type}")
    elif pretrained_ckpt is not None:
        print(f"\nLoading pretrained weights from {args.pretrained}")
        model.load_state_dict(pretrained_ckpt["model_state_dict"])
        pretrain_epoch = pretrained_ckpt.get("epoch", "?")
        pretrain_val_loss = pretrained_ckpt.get("val_loss", "?")
        ckpt_sched = pretrained_ckpt.get("scheduler_config", {})
        print(f"  Pretrained epoch: {pretrain_epoch}, val_loss: {pretrain_val_loss}")
        print(f"  Checkpoint scheduler_config: {ckpt_sched if ckpt_sched else '(none, legacy defaults)'}")
        print(f"  Resolved -> beta_schedule={beta_schedule}, prediction_type={prediction_type}")
        print("  Fresh optimizer and EMA (no state loaded)")

    # DataParallel
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SimpleMLP (context_dim={args.context_dim}, dropout={args.dropout})")
    print(f"  Total params: {param_count:,}")

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type=prediction_type,
    )

    # SNR diagnostics
    with torch.no_grad():
        test_ts = torch.tensor([0, 250, 500, 750, 999])
        ac = noise_scheduler.alphas_cumprod[test_ts]
        snr_vals = ac / (1.0 - ac)
        print(f"\nNoise schedule SNR diagnostics ({beta_schedule}):")
        for t, s in zip(test_ts.tolist(), snr_vals.tolist(), strict=True):
            print(f"  t={t:4d}: SNR={s:.4f} ({10 * math.log10(max(s, 1e-10)):.1f} dB)")

    # Build scheduler_config for checkpoint saving
    scheduler_config = {
        "beta_schedule": beta_schedule,
        "prediction_type": prediction_type,
        "scale_factor": args.scale_factor,
        "beta_start": 0.0015,
        "beta_end": 0.0195,
        "model_channels": 1536,
        "num_res_blocks": 12,
        "context_dim": args.context_dim,
    }

    # Optimizer and learning-rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, base_lr=args.lr, min_lr=args.min_lr
    )

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(model, decay=args.ema_decay)
        print(f"\nEMA initialized (decay={args.ema_decay})")

    # Restore state on resume
    if args.resume:
        ckpt_opt = ckpt.get("optimizer_state_dict")
        if ckpt_opt:
            optimizer.load_state_dict(ckpt_opt)
            print("  Restored optimizer state")
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
            print("  Restored EMA state")
        if "lr_scheduler_state_dict" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            except ValueError as error:
                parser.error(str(error))
            print("  Restored learning-rate scheduler state")
        else:
            for _ in range(start_epoch):
                lr_scheduler.step()
            print(f"  LR scheduler fast-forwarded to epoch {start_epoch} (legacy checkpoint)")
        if restore_rng_state(ckpt.get("rng_state")):
            print("  Restored PyTorch, CUDA, and NumPy RNG state")
        else:
            print("  Warning: checkpoint has no RNG state; resumed sampling will not be bitwise reproducible")

    # Save config
    manifest_hashes = {
        "train": sha256_file(args.manifest),
        "validation": sha256_file(args.val_manifest),
    }
    if args.replay_manifest:
        manifest_hashes["replay"] = sha256_file(args.replay_manifest)
    source_checkpoint = args.resume or args.pretrained

    config = vars(args).copy()
    config["effective_batch_size"] = eff_batch
    config["train_pairs_per_epoch"] = len(train_dataset)
    config["val_pairs_per_epoch"] = len(val_dataset)
    config["train_images"] = len(train_dataset.image_ids)
    config["val_images"] = len(val_dataset.image_ids)
    config["resolved_beta_schedule"] = beta_schedule
    config["resolved_prediction_type"] = prediction_type
    config["scheduler_config"] = scheduler_config
    config["manifest_sha256"] = manifest_hashes
    config["source_checkpoint_sha256"] = sha256_file(source_checkpoint) if source_checkpoint else None
    if replay_dataset:
        config["replay_samples"] = len(replay_dataset)
    if args.resume:
        previous_hashes = ckpt.get("training_config", {}).get("manifest_sha256")
        if previous_hashes and previous_hashes != manifest_hashes:
            parser.error(
                f"training manifest hashes do not match the resume checkpoint: "
                f"checkpoint={previous_hashes}, current={manifest_hashes}"
            )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    # Helpers to get state dictionaries
    def get_model_state():
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    def get_ema_state():
        if ema is not None:
            return ema.state_dict()
        return get_model_state()

    for key in (
        "train_loss",
        "val_loss",
        "train_image_loss",
        "train_text_loss",
        "train_replay_loss",
        "val_image_loss",
        "val_text_loss",
        "lr",
    ):
        history.setdefault(key, [])

    def make_checkpoint(epoch, train_metrics, val_metrics, rng_state):
        raw_model_state = get_model_state()
        inference_state = get_ema_state()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": inference_state,
            "raw_model_state_dict": raw_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "best_val_loss": best_val_loss,
            "context_dim": args.context_dim,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "scheduler_config": scheduler_config,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "rng_state": rng_state,
            "training_config": config,
            "history": history,
        }
        if ema is not None:
            checkpoint["ema_state_dict"] = inference_state
        return checkpoint

    def save_history():
        history_path = output_dir / "training_history.json"
        temporary = history_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(history_path)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = lr_scheduler.get_lr()
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}  (lr={lr:.2e}, eff_batch={eff_batch})")
        print(f"{'=' * 70}")

        train_dataset.resample()

        train_metrics = train_epoch(
            model,
            train_loader,
            replay_loader,
            optimizer,
            noise_scheduler,
            device,
            scale_factor=args.scale_factor,
            unconditional_prob=args.unconditional_prob,
            max_grad_norm=args.max_grad_norm,
            grad_accum=args.grad_accum,
            replay_ratio=args.replay_ratio if replay_loader else 0,
            snr_gamma=args.snr_gamma,
            ema=ema,
        )

        # Validate with EMA weights if available
        val_dataset.resample()
        if ema is not None:
            ema.apply_shadow(model)

        val_metrics = validate(
            model,
            val_loader,
            noise_scheduler,
            device,
            scale_factor=args.scale_factor,
            snr_gamma=args.snr_gamma,
        )

        if ema is not None:
            ema.restore(model)

        lr_scheduler.step()

        replay_str = ""
        if train_metrics["replay_count"] > 0:
            replay_str = f", replay={train_metrics['replay_loss']:.4f} [{train_metrics['replay_count']}]"
        print(
            f"Train: loss={train_metrics['loss']:.4f} "
            f"(img={train_metrics['image_loss']:.4f} [{train_metrics['image_count']}], "
            f"txt={train_metrics['text_loss']:.4f} [{train_metrics['text_count']}]"
            f"{replay_str})"
        )
        print(
            f"Val:   loss={val_metrics['loss']:.4f} "
            f"(img={val_metrics['image_loss']:.4f}, txt={val_metrics['text_loss']:.4f})"
        )
        print(f"  Optimizer steps: {train_metrics['optimizer_steps']}")

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_image_loss"].append(train_metrics["image_loss"])
        history["train_text_loss"].append(train_metrics["text_loss"])
        history["train_replay_loss"].append(train_metrics.get("replay_loss", 0))
        history["val_image_loss"].append(val_metrics["image_loss"])
        history["val_text_loss"].append(val_metrics["text_loss"])
        history["lr"].append(lr)

        val_loss = val_metrics["loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        history["best_val_loss"] = best_val_loss
        history["context_dim"] = args.context_dim
        history["scheduler_config"] = scheduler_config

        save_periodic = (epoch + 1) % args.save_every == 0
        if is_best or save_periodic:
            rng_state = capture_rng_state()
            save_dict = make_checkpoint(epoch + 1, train_metrics, val_metrics, rng_state)

        if is_best:
            atomic_torch_save(save_dict, output_dir / "best_model.pth")
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

        if save_periodic:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:04d}.pth"
            atomic_torch_save(save_dict, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path.name}")

        save_history()

    print(f"\n{'=' * 70}")
    print("Fine-tuning complete!")
    print(f"  Noise schedule:  {beta_schedule}")
    print(f"  Prediction type: {prediction_type}")
    print(f"  Best val loss:   {best_val_loss:.6f}")
    print(f"  Best model:      {output_dir / 'best_model.pth'}")
    print(f"  History:         {output_dir / 'training_history.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
