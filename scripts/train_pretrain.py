#!/usr/bin/env python3
"""Pretrain the Meric Music Head from validated cached embeddings.

The camera-ready recipe maps Qwen3-VL conditions to MuQ targets with a
squared-cosine noise schedule, v-prediction, EMA, min-SNR weighting, and a
warmup-plus-cosine learning-rate schedule. See ``docs/REPRODUCIBILITY.md`` for
the canonical invocation and the current data-release boundary.
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

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from meric.data.unified_dataset import UnifiedEmbeddingDataset  # noqa: E402
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

# Auto-detect context_dim from cond_key
CONTEXT_DIM_MAP = {
    "clip": 1024,
    "clip_keyframes": 1024,
    "qwen3vl": 2048,
    "qwen3vl_keyframes": 2048,
}


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
    optimizer,
    noise_scheduler,
    device,
    scale_factor=10.0,
    unconditional_prob=0.1,
    max_grad_norm=1.0,
    grad_accum=1,
    snr_gamma=5.0,
    ema=None,
):
    """Train for one epoch and return sample-weighted loss and optimizer steps."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    n_steps = 0
    accum_count = 0
    accum_samples = 0

    optimizer.zero_grad()

    for clip_batch, muq_batch in tqdm(train_loader, desc="Training"):
        clip_batch = clip_batch.to(device)
        muq_batch = muq_batch.to(device)
        batch_size = clip_batch.shape[0]

        # Scale MuQ features (target)
        audio_feat = muq_batch * scale_factor

        # Sample noise and timesteps
        noise = torch.randn_like(audio_feat)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()

        # Add noise to MuQ features
        noisy_audio_feat = noise_scheduler.add_noise(audio_feat, noise, timesteps)

        # Conditional dropout (classifier-free guidance training)
        image_features = clip_batch.clone()
        if unconditional_prob > 0:
            mask = torch.rand(batch_size, 1, device=device) < unconditional_prob
            image_features = image_features * (~mask).float()

        # Get prediction target
        target = get_prediction_target(noise_scheduler, audio_feat, noise, timesteps)

        # Forward pass
        pred = model(x=noisy_audio_feat, timesteps=timesteps, context=image_features)

        # Per-sample MSE loss
        per_sample_loss = nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)

        # Min-SNR weighting
        if snr_gamma > 0:
            snr = compute_snr(noise_scheduler, timesteps)
            snr_weights = compute_snr_weights(snr, gamma=snr_gamma)
            per_sample_loss = per_sample_loss * snr_weights

        if not torch.isfinite(per_sample_loss).all():
            raise FloatingPointError("Training loss became non-finite")

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

            # Update EMA after each optimizer step
            if ema is not None:
                ema.update(model)

        total_loss += per_sample_loss.sum().item()
        total_samples += batch_size

    # Handle remaining accumulated gradients
    if accum_count > 0:
        normalize_gradients(model.parameters(), accum_samples)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1
        if ema is not None:
            ema.update(model)

    if total_samples < 1:
        raise RuntimeError("Training dataloader produced no samples")
    avg_loss = total_loss / total_samples
    return avg_loss, n_steps


def validate(model, val_loader, noise_scheduler, device, scale_factor=10.0, snr_gamma=5.0):
    """Validate the model (call with EMA weights applied)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for clip_batch, muq_batch in val_loader:
            clip_batch = clip_batch.to(device)
            muq_batch = muq_batch.to(device)
            batch_size = clip_batch.shape[0]

            # Scale MuQ features (target)
            audio_feat = muq_batch * scale_factor

            # Sample noise and timesteps
            noise = torch.randn_like(audio_feat)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
            timesteps = timesteps.long()

            # Add noise
            noisy_audio_feat = noise_scheduler.add_noise(audio_feat, noise, timesteps)

            # Get prediction target
            target = get_prediction_target(noise_scheduler, audio_feat, noise, timesteps)

            # Forward pass
            pred = model(x=noisy_audio_feat, timesteps=timesteps, context=clip_batch)

            # Per-sample MSE with Min-SNR weighting
            per_sample_loss = nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)

            if snr_gamma > 0:
                snr = compute_snr(noise_scheduler, timesteps)
                snr_weights = compute_snr_weights(snr, gamma=snr_gamma)
                per_sample_loss = per_sample_loss * snr_weights

            if not torch.isfinite(per_sample_loss).all():
                raise FloatingPointError("Validation loss became non-finite")
            total_loss += per_sample_loss.sum().item()
            total_samples += batch_size

    if total_samples < 1:
        raise RuntimeError("Validation dataloader produced no samples")
    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, default="data/unified/meta/visual_train.jsonl", help="Training JSONL manifest"
    )
    parser.add_argument(
        "--val-manifest",
        "--val_manifest",
        type=str,
        default="data/unified/meta/test_audioset_music.jsonl",
        help="Validation JSONL manifest",
    )
    parser.add_argument(
        "--cond-key",
        "--cond_key",
        type=str,
        default="qwen3vl",
        choices=["clip", "clip_keyframes", "qwen3vl", "qwen3vl_keyframes"],
        help="Condition embedding key (default: qwen3vl)",
    )
    parser.add_argument(
        "--target-key", "--target_key", type=str, default="muq", help="Target embedding key (default: muq)"
    )
    parser.add_argument(
        "--filter-datasets",
        "--filter_datasets",
        type=str,
        nargs="*",
        default=None,
        help="Filter to these dataset names (e.g., audioset_music muimage)",
    )
    parser.add_argument(
        "--data-root", "--data_root", type=str, default=".", help="Root dir for resolving relative paths in manifest"
    )
    parser.add_argument("--output-dir", "--output_dir", type=str, default="outputs/checkpoints/rdm_pretrain")
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Init weights from existing checkpoint (fresh optimizer/EMA)"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scale-factor", "--scale_factor", type=float, default=10.0)
    parser.add_argument("--unconditional-prob", "--unconditional_prob", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate in ResBlocks (default: 0.05)")
    parser.add_argument(
        "--max-grad-norm",
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
    )
    parser.add_argument(
        "--grad-accum",
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch size * accumulation steps)",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=8,
        help="Worker processes per dataloader (default: 8)",
    )
    parser.add_argument(
        "--preload-workers",
        "--preload_workers",
        type=int,
        default=32,
        help="Threads used to validate and preload cached embeddings (default: 32)",
    )

    # LR schedule
    parser.add_argument(
        "--warmup-epochs", "--warmup_epochs", type=int, default=5, help="Linear warmup epochs before cosine decay"
    )
    parser.add_argument("--min-lr", "--min_lr", type=float, default=1e-6, help="Minimum LR for cosine decay")

    # EMA
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")

    # Noise schedule and prediction type
    parser.add_argument(
        "--beta-schedule",
        "--beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        choices=["linear", "squaredcos_cap_v2", "scaled_linear"],
        help="Noise schedule (default: squaredcos_cap_v2 = cosine)",
    )
    parser.add_argument(
        "--prediction-type",
        "--prediction_type",
        type=str,
        default="v_prediction",
        choices=["sample", "epsilon", "v_prediction"],
        help="Prediction target (default: v_prediction)",
    )

    # Min-SNR
    parser.add_argument(
        "--snr-gamma", "--snr_gamma", type=float, default=5.0, help="Min-SNR gamma for loss weighting (0 to disable)"
    )

    # Model architecture
    parser.add_argument(
        "--model-channels", "--model_channels", type=int, default=1536, help="Model channel width (default: 1536)"
    )
    parser.add_argument(
        "--num-res-blocks", "--num_res_blocks", type=int, default=12, help="Number of residual blocks (default: 12)"
    )

    # Checkpointing
    parser.add_argument("--save-every", "--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
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
    if not 0 <= args.unconditional_prob <= 1:
        parser.error("--unconditional-prob must be between 0 and 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be at least 0 and less than 1")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be at least 0 and less than 1")
    if args.scale_factor <= 0 or args.weight_decay < 0 or args.max_grad_norm < 0 or args.snr_gamma < 0:
        parser.error("scale must be positive; decay, clipping, and SNR values must be non-negative")
    if args.model_channels < 2 or args.num_res_blocks < 1:
        parser.error("--model-channels must be at least 2 and --num-res-blocks must be at least 1")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32 - 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
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

    # Auto-detect context_dim
    context_dim = CONTEXT_DIM_MAP.get(args.cond_key, 1024)
    eff_batch = args.batch_size * args.grad_accum

    print("=" * 70)
    print("Training RDM with Cached Features (Optimized)")
    print("=" * 70)
    print(f"Manifest:       {args.manifest}")
    print(f"Val manifest:   {args.val_manifest}")
    print(f"Cond key:       {args.cond_key} (context_dim={context_dim})")
    print(f"Target key:     {args.target_key}")
    print(f"Filter:         {args.filter_datasets}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Pretrained:     {args.pretrained}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size} × {args.grad_accum} accum = {eff_batch} effective")
    print(f"Learning rate:  {args.lr} (warmup {args.warmup_epochs} epochs → cosine to {args.min_lr})")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Grad clip:      {args.max_grad_norm}")
    print(f"Dropout:        {args.dropout}")
    print(f"EMA decay:      {args.ema_decay}")
    print(f"Noise schedule: {args.beta_schedule}")
    print(f"Prediction:     {args.prediction_type}")
    print(f"Min-SNR gamma:  {args.snr_gamma}")
    print(f"Model channels: {args.model_channels}")
    print(f"Num res blocks: {args.num_res_blocks}")
    print(f"Devices:        {device_ids}")
    print(f"Scale factor:   {args.scale_factor}")
    print(f"Data workers:   {args.num_workers} process(es), {args.preload_workers} preload thread(s)")

    # Load data from manifests
    print("\nLoading training data...")
    train_dataset = UnifiedEmbeddingDataset(
        args.manifest,
        cond_key=args.cond_key,
        target_key=args.target_key,
        preload=True,
        data_root=args.data_root,
        filter_datasets=args.filter_datasets,
        preload_workers=args.preload_workers,
    )

    print("\nLoading validation data...")
    val_dataset = UnifiedEmbeddingDataset(
        args.val_manifest,
        cond_key=args.cond_key,
        target_key=args.target_key,
        preload=True,
        data_root=args.data_root,
        preload_workers=args.preload_workers,
    )

    # Create dataloaders
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

    print("\nData split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Dataset breakdown (train): {train_dataset.dataset_counts}")

    # Create model with dropout
    model = SimpleMLP(
        in_channels=512,  # MuQ dim
        time_embed_dim=256,
        model_channels=args.model_channels,
        bottleneck_channels=args.model_channels,
        out_channels=512,  # MuQ dim
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        use_context=True,
        context_channels=context_dim,
    )

    # Load pretrained weights (fresh optimizer, fresh EMA)
    if args.pretrained and not args.resume:
        print(f"\nLoading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        pretrain_epoch = checkpoint.get("epoch", "?")
        pretrain_val_loss = checkpoint.get("val_loss", "?")
        print(f"  Pretrained epoch: {pretrain_epoch}, val_loss: {pretrain_val_loss}")
        print("  Fresh optimizer and EMA (no state loaded)")

    # Use DataParallel if multiple GPUs
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device)

    print("\nRDM architecture:")
    print(f"  Context ({args.cond_key}): {context_dim}")
    print("  Input/Output (MuQ): 512")
    print(f"  Model channels: {args.model_channels}")
    print(f"  Res blocks: {args.num_res_blocks}")
    print(f"  Dropout: {args.dropout}")
    base_model = model.module if hasattr(model, "module") else model
    print(f"  Total params: {sum(p.numel() for p in base_model.parameters()):,}")

    # Create noise scheduler
    # For squaredcos_cap_v2, beta_start/beta_end are ignored (schedule derived from cosine)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
        prediction_type=args.prediction_type,
    )

    # Print SNR diagnostics
    with torch.no_grad():
        test_ts = torch.tensor([0, 250, 500, 750, 999])
        ac = noise_scheduler.alphas_cumprod[test_ts]
        snr_vals = ac / (1.0 - ac)
        print(f"\nNoise schedule SNR diagnostics ({args.beta_schedule}):")
        for t, s in zip(test_ts.tolist(), snr_vals.tolist(), strict=True):
            print(f"  t={t:4d}: SNR={s:.4f} ({10 * math.log10(max(s, 1e-10)):.1f} dB)")

    # Create optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler: warmup + cosine decay
    lr_scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, base_lr=args.lr, min_lr=args.min_lr
    )

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(model, decay=args.ema_decay)
        print(f"\nEMA initialized (decay={args.ema_decay})")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    history = {
        "train_losses": [],
        "val_losses": [],
        "lrs": [],
        "best_val_loss": best_val_loss,
    }

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)

        checkpoint_scheduler = checkpoint.get("scheduler_config", {})
        expected_scheduler = {
            "beta_schedule": args.beta_schedule,
            "prediction_type": args.prediction_type,
            "scale_factor": args.scale_factor,
            "model_channels": args.model_channels,
            "num_res_blocks": args.num_res_blocks,
            "context_dim": context_dim,
        }
        mismatches = {
            key: (checkpoint_scheduler[key], value)
            for key, value in expected_scheduler.items()
            if key in checkpoint_scheduler and checkpoint_scheduler[key] != value
        }
        if mismatches:
            parser.error(f"resume checkpoint configuration does not match this run: {mismatches}")

        previous_config = checkpoint.get("training_config", {})
        exact_resume_fields = (
            "cond_key",
            "target_key",
            "filter_datasets",
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
            "model_channels",
            "num_res_blocks",
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

        # Load raw model weights (not EMA) for continued training
        raw_state = checkpoint.get("raw_model_state_dict", checkpoint["model_state_dict"])
        if hasattr(model, "module"):
            model.module.load_state_dict(raw_state)
        else:
            model.load_state_dict(raw_state)

        # Restore optimizer
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  Restored optimizer state")

        # Restore EMA
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            print("  Restored EMA state")

        start_epoch = checkpoint["epoch"]
        if not isinstance(start_epoch, int) or start_epoch < 0:
            parser.error(f"resume checkpoint has an invalid epoch: {start_epoch!r}")
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "lr_scheduler_state_dict" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            except ValueError as error:
                parser.error(str(error))
            print("  Restored learning-rate scheduler state")
        else:
            for _ in range(start_epoch):
                lr_scheduler.step()
            print(f"  LR scheduler fast-forwarded to epoch {start_epoch} (legacy checkpoint)")

        history_path = output_dir / "training_history.json"
        if "history" in checkpoint:
            history = checkpoint["history"]
        elif history_path.is_file():
            with history_path.open(encoding="utf-8") as handle:
                history = json.load(handle)

        if restore_rng_state(checkpoint.get("rng_state")):
            print("  Restored PyTorch, CUDA, and NumPy RNG state")
        else:
            print("  Warning: checkpoint has no RNG state; resumed sampling will not be bitwise reproducible")
        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        if start_epoch >= args.epochs:
            parser.error(f"checkpoint epoch {start_epoch} is not less than --epochs {args.epochs}")

    # Build scheduler_config for checkpoint saving
    scheduler_config = {
        "beta_schedule": args.beta_schedule,
        "prediction_type": args.prediction_type,
        "scale_factor": args.scale_factor,
        "beta_start": 0.0015,
        "beta_end": 0.0195,
        "model_channels": args.model_channels,
        "num_res_blocks": args.num_res_blocks,
        "context_dim": context_dim,
    }

    manifest_hashes = {
        "train": sha256_file(args.manifest),
        "validation": sha256_file(args.val_manifest),
    }
    source_checkpoint = args.resume or args.pretrained

    training_config = vars(args).copy()
    training_config.update(
        {
            "context_dim": context_dim,
            "effective_batch_size": eff_batch,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "scheduler_config": scheduler_config,
            "manifest_sha256": manifest_hashes,
            "source_checkpoint_sha256": sha256_file(source_checkpoint) if source_checkpoint else None,
        }
    )
    if args.resume:
        previous_hashes = checkpoint.get("training_config", {}).get("manifest_sha256")
        if previous_hashes and previous_hashes != manifest_hashes:
            parser.error(
                f"training manifest hashes do not match the resume checkpoint: "
                f"checkpoint={previous_hashes}, current={manifest_hashes}"
            )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(training_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    train_losses = history.setdefault("train_losses", [])
    val_losses = history.setdefault("val_losses", [])
    lrs = history.setdefault("lrs", [])

    def get_model_state():
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    def get_ema_state():
        if ema is not None:
            return ema.state_dict()
        return get_model_state()

    def make_checkpoint(epoch, train_loss, val_loss, rng_state):
        raw_model_state = get_model_state()
        inference_state = get_ema_state()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": inference_state,
            "raw_model_state_dict": raw_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "scheduler_config": scheduler_config,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "rng_state": rng_state,
            "training_config": training_config,
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

    for epoch in range(start_epoch, args.epochs):
        lr = lr_scheduler.get_lr()
        lrs.append(lr)

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}  (lr={lr:.2e}, eff_batch={eff_batch})")
        print(f"{'=' * 70}")

        # Train
        train_loss, n_steps = train_epoch(
            model,
            train_loader,
            optimizer,
            noise_scheduler,
            device,
            scale_factor=args.scale_factor,
            unconditional_prob=args.unconditional_prob,
            max_grad_norm=args.max_grad_norm,
            grad_accum=args.grad_accum,
            snr_gamma=args.snr_gamma,
            ema=ema,
        )
        train_losses.append(train_loss)

        # Step LR scheduler (after training epoch)
        lr_scheduler.step()

        # Validate with EMA weights (if available)
        if ema is not None:
            ema.apply_shadow(model)

        val_loss = validate(
            model, val_loader, noise_scheduler, device, scale_factor=args.scale_factor, snr_gamma=args.snr_gamma
        )
        val_losses.append(val_loss)

        if ema is not None:
            ema.restore(model)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Steps: {n_steps}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        history["best_val_loss"] = best_val_loss
        history["scheduler_config"] = scheduler_config
        save_periodic = (epoch + 1) % args.save_every == 0
        if is_best or save_periodic:
            rng_state = capture_rng_state()
            save_dict = make_checkpoint(epoch + 1, train_loss, val_loss, rng_state)

        if is_best:
            atomic_torch_save(save_dict, output_dir / "best_model.pth")
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

        if save_periodic:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:04d}.pth"
            atomic_torch_save(save_dict, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path.name}")

        save_history()

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {output_dir / 'best_model.pth'}")
    print(f"  History: {output_dir / 'training_history.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
