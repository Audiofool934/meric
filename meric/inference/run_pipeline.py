#!/usr/bin/env python3
"""Low-level Meric inference pipeline.

Supports multiple input modes and vision encoders:
  - Image → CLIP RDM → MuQ → MericLDM → Audio
  - Image → Qwen3-VL RDM → MuQ → MericLDM → Audio
  - Video → Qwen3-VL RDM → MuQ → MericLDM → Audio
  - Text → Qwen3-VL RDM → MuQ → MericLDM → Audio
  - MuQ embedding (npy) → MericLDM → Audio

Usage examples:

  # Single image with the released Qwen3-VL backbone
  python -m meric.inference.run_pipeline \
      --image path/to/photo.jpg \
      --output-dir outputs/my_test

  # Directory of images
  python -m meric.inference.run_pipeline \
      --image path/to/image_folder/ \
      --output-dir outputs/batch_test

  # Qwen3-VL backbone
  python -m meric.inference.run_pipeline \
      --image path/to/photo.jpg \
      --backbone qwen3vl \
      --rdm-path outputs/checkpoints/rdm_pretrain/best_model.pth \
      --output-dir outputs/qwen_test

  # Text prompt through the paper's Qwen3-VL Music Head path
  python -m meric.inference.run_pipeline \
      --text "A calm piano melody with gentle rain" \
      --output-dir outputs/text_test

  # Video through the same multimodal Music Head
  python -m meric.inference.run_pipeline \
      --video path/to/video.mp4 \
      --output-dir outputs/video_test

  # Pre-extracted MuQ embedding (bypasses Stage 1)
  python -m meric.inference.run_pipeline \
      --muq-npy path/to/muq_embedding.npy \
      --output-dir outputs/muq_test

  # Generate 3 random variants per image
  python -m meric.inference.run_pipeline \
      --image photo.jpg -n 3 \
      --output-dir outputs/diverse

  # Or specify exact seeds
  python -m meric.inference.run_pipeline \
      --image photo.jpg --seeds 42,123,456 \
      --output-dir outputs/diverse

  # Adjust generation parameters
  python -m meric.inference.run_pipeline \
      --image photo.jpg \
      --guidance-scale 7.0 \
      --sample-steps 100 \
      --rdm-guidance 3.0 \
      --output-dir outputs/high_quality
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ============================================================
# Multimodal Encoders
# ============================================================

VISUAL_INSTRUCTION = "Represent the emotional atmosphere and mood of this visual content for music generation."
TEXT_INSTRUCTION = "Represent the emotional atmosphere and mood of this description for music generation."
_MAX_SEED = 2**63 - 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_embedding_file(path, expected_dim, label):
    """Load one numeric embedding and normalize it to shape ``[1, D]``."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        loaded = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load {label} file {path}: {error}") from error
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"{label} file must contain one NumPy array: {path}")
    if not np.issubdtype(loaded.dtype, np.number) or np.issubdtype(loaded.dtype, np.complexfloating):
        raise ValueError(f"{label} file must contain a real numeric array: {path}")
    try:
        array = np.asarray(loaded, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} file must contain a numeric array: {path}") from error
    if array.ndim == 1:
        array = array[None, :]
    if array.shape != (1, expected_dim):
        raise ValueError(
            f"Expected one {label} embedding with shape ({expected_dim},) or (1, {expected_dim}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} embedding contains non-finite values")
    return array


def _safe_name(value, fallback):
    name = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("_.")
    return name[:64] or fallback


def load_clip_encoder(device):
    """Load OpenCLIP ViT-H-14."""
    import open_clip

    print("Loading CLIP ViT-H-14 (laion2b_s32b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model = model.to(device).eval()
    print("  CLIP loaded. Output dim: 1024")
    return model, preprocess


def extract_clip_embedding(model, preprocess, image_path, device):
    """Extract CLIP embedding from a single image."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)  # [1, 1024]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


def _qwen3vl_runtime() -> tuple[Path, Path]:
    """Resolve and validate the external Qwen3-VL repository and Python."""
    default_repo = Path(__file__).resolve().parents[2] / ".external" / "Qwen3-VL-Embedding"
    repo = Path(os.environ.get("QWEN3VL_REPO", default_repo)).expanduser().resolve()
    python = repo / ".venv" / "bin" / "python"
    if not repo.is_dir():
        raise FileNotFoundError(
            f"QWEN3VL_REPO does not exist: {repo}. Follow docs/SETUP.md to install the pinned backbone."
        )
    if not python.is_file():
        raise FileNotFoundError(f"Qwen3-VL Python environment not found: {python}. Follow docs/SETUP.md.")
    from meric import hub

    revision = None
    if shutil.which("git"):
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    actual_revision = revision.stdout.strip() if revision and revision.returncode == 0 else None
    if actual_revision and actual_revision != hub.QWEN3VL_CODE_REVISION:
        warnings.warn(
            "QWEN3VL_REPO is not at the tested revision "
            f"{hub.QWEN3VL_CODE_REVISION}; found {actual_revision}. Results may differ.",
            stacklevel=2,
        )
    return repo, python


def _qwen_cuda_visible_devices(device_str: str, current_value: str | None) -> str | None:
    """Map a parent-process CUDA device to one visible Qwen subprocess GPU."""
    if device_str == "cpu":
        return ""
    if device_str == "cuda":
        return current_value
    if not device_str.startswith("cuda:"):
        return current_value

    logical_index = int(device_str.split(":", 1)[1])
    if logical_index < 0:
        raise ValueError(f"Invalid CUDA device: {device_str}")
    if not current_value:
        return str(logical_index)

    visible_devices = [value.strip() for value in current_value.split(",") if value.strip()]
    if logical_index >= len(visible_devices):
        raise ValueError(
            f"{device_str} is outside CUDA_VISIBLE_DEVICES={current_value!r}; "
            f"choose cuda:0 through cuda:{len(visible_devices) - 1}"
        )
    return visible_devices[logical_index]


def _extract_qwen3vl_inputs_via_subprocess(inputs, output_dir, device_str):
    """Extract normalized Qwen3-VL embeddings in its isolated environment."""
    if not inputs:
        raise ValueError("At least one Qwen3-VL input is required")

    repo, python = _qwen3vl_runtime()
    helper_script = Path(__file__).with_name("_extract_qwen3vl.py").resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    request_handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="meric_qwen3vl_", dir=output_dir, delete=False, encoding="utf-8"
    )
    result_handle = tempfile.NamedTemporaryFile(suffix=".npy", prefix="meric_qwen3vl_", dir=output_dir, delete=False)
    request_path = Path(request_handle.name)
    result_path = Path(result_handle.name)
    result_handle.close()
    result_path.unlink(missing_ok=True)

    try:
        with request_handle:
            json.dump({"inputs": inputs}, request_handle, ensure_ascii=False)

        env = os.environ.copy()
        env["QWEN3VL_REPO"] = str(repo)
        visible_device = _qwen_cuda_visible_devices(device_str, env.get("CUDA_VISIBLE_DEVICES"))
        if visible_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_device

        print(f"Extracting Qwen3-VL embeddings via {python}...")
        result = subprocess.run(
            [
                str(python),
                str(helper_script),
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ],
            capture_output=True,
            text=True,
            cwd=repo,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise RuntimeError(f"Qwen3-VL extraction failed:\n{detail}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"  [qwen3vl] {line}")

        embeddings = np.load(result_path, allow_pickle=False)
        expected_shape = (len(inputs), 2048)
        if embeddings.shape != expected_shape:
            raise ValueError(f"Expected Qwen3-VL embeddings {expected_shape}, got {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("Qwen3-VL produced non-finite embeddings")
        return embeddings
    finally:
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _extract_qwen3vl_via_subprocess(images, output_dir, device_str):
    """Extract image embeddings and return them keyed by resolved path."""
    resolved = [Path(image).resolve() for image in images]
    inputs = [{"image": str(image), "instruction": VISUAL_INSTRUCTION} for image in resolved]
    embeddings = _extract_qwen3vl_inputs_via_subprocess(inputs, output_dir, device_str)
    return {str(image): embeddings[index : index + 1] for index, image in enumerate(resolved)}


def _extract_qwen3vl_text_via_subprocess(texts, output_dir, device_str):
    inputs = [{"text": text, "instruction": TEXT_INSTRUCTION} for text in texts]
    return _extract_qwen3vl_inputs_via_subprocess(inputs, output_dir, device_str)


def _extract_qwen3vl_video_via_subprocess(videos, output_dir, device_str, max_frames=8):
    resolved = [Path(video).resolve() for video in videos]
    inputs = [
        {
            "video": str(video),
            "instruction": VISUAL_INSTRUCTION,
            "max_frames": max_frames,
        }
        for video in resolved
    ]
    embeddings = _extract_qwen3vl_inputs_via_subprocess(inputs, output_dir, device_str)
    return {str(video): embeddings[index : index + 1] for index, video in enumerate(resolved)}


# ============================================================
# Stage 1: Vision/Text → MuQ via RDM
# ============================================================


def load_rdm(rdm_path, context_dim, device):
    """Load a trained RDM Music Head and its DDIM scheduler."""
    import torch as _torch

    checkpoint = _torch.load(rdm_path, map_location="cpu", weights_only=True, mmap=True)

    from meric.utils.rdm_utils import load_rdm_checkpoint

    print(f"Loading RDM Music Head from {rdm_path}...")
    model, scheduler, scale_factor = load_rdm_checkpoint(rdm_path, context_dim, device, checkpoint=checkpoint)
    scheduler._rdm_scale_factor = scale_factor
    return model, scheduler


@torch.no_grad()
def rdm_generate_muq(
    model, scheduler, vision_emb, device, num_steps=20, guidance_scale=2.0, scale_factor=10.0, seed=42
):
    """Sample a MuQ anchor from one or more Stage-1 conditions with DDIM."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite")
    from meric.utils.rdm_utils import batch_ddim_sample

    result = batch_ddim_sample(
        model,
        vision_emb,
        scheduler,
        device,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        scale_factor=scale_factor,
        seed=seed,
        batch_size=vision_emb.shape[0],
    )
    result = torch.tensor(result, dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] != 512 or not torch.isfinite(result).all():
        raise RuntimeError(f"RDM returned an invalid MuQ tensor with shape {tuple(result.shape)}")
    return result


# ============================================================
# Stage 2: MuQ → Audio via MericLDM
# ============================================================


def load_meric_ldm(
    device,
    stage2_ckpt,
    guidance_scale=5.0,
    sample_steps=50,
    sample_method="dopri5",
    vae_dir=None,
    transformer_dir=None,
    muq_dir=None,
):
    """Load MericLDM for audio generation.

    Component directories resolve through the pinned Hub registry unless they
    are passed explicitly.

    Stage 2 is conditioned by MuQ-MuLan, not CLAP.
    """
    from meric.workers.stableaudio_muq_flow import MericLDM

    if vae_dir is None or transformer_dir is None:
        from meric import hub

        resolved_vae, resolved_transformer = hub.resolve_stable_audio()
        vae_dir = vae_dir or resolved_vae
        transformer_dir = transformer_dir or resolved_transformer
    if muq_dir is None:
        from meric import hub

        muq_dir = hub.resolve_muq_mulan()

    print("Loading MericLDM (Stage 2)...")
    model = (
        MericLDM(
            audiocodec_ckpt_path=vae_dir,
            ckpt_dir_audio_dit=transformer_dir,
            muq_model_name_or_path=muq_dir,
            meric_ckpt_path=stage2_ckpt,
            cond_feat_dim=512,
            use_cache_audio_feat=True,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            sample_method=sample_method,
        )
        .to(device)
        .eval()
    )
    print(f"  MericLDM loaded (steps={sample_steps}, cfg={guidance_scale}, method={sample_method})")
    return model


def generate_audio(meric_model, muq_emb, output_dir, name, device, seed=42):
    """Generate audio from MuQ embedding and save to WAV."""
    muq_tensor = (
        muq_emb.to(device)
        if isinstance(muq_emb, torch.Tensor)
        else torch.tensor(muq_emb, dtype=torch.float32).to(device)
    )

    if muq_tensor.dim() == 1:
        muq_tensor = muq_tensor.unsqueeze(0)

    with torch.no_grad():
        generated = meric_model.generate_waveforms(muq_tensor, seeds=[seed])
    if len(generated) != 1 or generated[0].ndim != 3 or generated[0].shape[0] != 1:
        raise RuntimeError("Stage 2 returned an unexpected waveform batch")

    # Peak-normalize audio to consistent volume.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = Path(output_dir) / f"{name}_00.wav"
    import torchaudio

    audio = generated[0][0].detach().cpu()
    if audio.ndim != 2 or audio.shape[-1] < 1 or not torch.isfinite(audio).all():
        raise RuntimeError(f"Stage 2 returned an invalid waveform with shape {tuple(audio.shape)}")
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95
    torchaudio.save(str(wav_path), audio, meric_model.audio_sample_rate)

    return wav_path


# ============================================================
# Video Synthesis
# ============================================================


def make_video(image_path, wav_path, video_path):
    """Combine a static image with audio into an MP4 video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(wav_path),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-shortest",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.splitlines()[-1] if result.stderr else 'unknown'}")
        return None
    return video_path


def make_concat_video(video_paths, output_path):
    """Concatenate multiple videos sequentially (for multi-seed comparison)."""
    if len(video_paths) < 2:
        return None
    # Create concat list file with absolute paths
    list_file = output_path.parent / f"_concat_{output_path.stem}.txt"
    with open(list_file, "w") as f:
        for vp in video_paths:
            f.write(f"file '{Path(vp).resolve()}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  ffmpeg concat error: {result.stderr.splitlines()[-1] if result.stderr else 'unknown'}")
        return None
    return output_path


# ============================================================
# Main
# ============================================================


def collect_images(path):
    """Collect image paths from a file or directory."""
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        images = []
        for ext in exts:
            images.extend(sorted(p.glob(ext)))
        return images
    else:
        raise FileNotFoundError(f"Not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Meric low-level inference: image, video, text, or MuQ to music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Image file or directory of images")
    input_group.add_argument("--video", type=str, help="Video file")
    input_group.add_argument("--text", type=str, help="Text prompt")
    input_group.add_argument(
        "--muq-npy", "--muq_npy", dest="muq_npy", type=str, help="Pre-extracted MuQ embedding (.npy, skips Stage 1)"
    )
    input_group.add_argument(
        "--vision-npy",
        "--vision_npy",
        dest="vision_npy",
        type=str,
        help="Pre-extracted vision embedding (.npy). Use with --backbone to select RDM. "
        "Required for qwen3vl (extract embeddings in qwen3 env first)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="outputs/inference",
        help="Output directory for WAV files",
    )

    # Backbone selection
    parser.add_argument(
        "--backbone",
        type=str,
        default="qwen3vl",
        choices=["clip", "qwen3vl"],
        help="Stage-1 backbone (default: qwen3vl)",
    )
    parser.add_argument(
        "--video-max-frames", type=int, default=8, help="Maximum Qwen3-VL frames sampled from video input"
    )

    # RDM (Stage 1) settings
    parser.add_argument(
        "--rdm-path",
        "--rdm_path",
        dest="rdm_path",
        type=str,
        default=None,
        help="RDM checkpoint path (auto-detected from backbone if not set)",
    )
    parser.add_argument(
        "--rdm-steps", "--rdm_steps", dest="rdm_steps", type=int, default=20, help="RDM denoising steps"
    )
    parser.add_argument(
        "--rdm-guidance", "--rdm_guidance", dest="rdm_guidance", type=float, default=2.0, help="RDM CFG scale"
    )
    parser.add_argument("--rdm-scale-factor", "--rdm_scale_factor", dest="rdm_scale_factor", type=float, default=10.0)
    parser.add_argument(
        "--rdm-ensemble",
        "--rdm_ensemble",
        dest="rdm_ensemble",
        type=int,
        default=1,
        help="Ensemble N RDM predictions by averaging MuQ (e.g. 10). "
        "Uses seeds 0..N-1 for ensemble, then applies --seeds for Stage 2.",
    )

    # MericLDM (Stage 2) settings
    parser.add_argument("--stage2-ckpt", "--stage2_ckpt", dest="stage2_ckpt", type=str, default=None)
    parser.add_argument(
        "--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=5.0, help="Stage 2 CFG scale"
    )
    parser.add_argument(
        "--sample-steps", "--sample_steps", dest="sample_steps", type=int, default=50, help="Stage 2 ODE solver steps"
    )
    parser.add_argument(
        "--sample-method",
        "--sample_method",
        dest="sample_method",
        type=str,
        default="dopri5",
        choices=["euler", "midpoint", "dopri5"],
    )

    # General
    parser.add_argument("--device", type=str, default="cuda:0")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (e.g. 42,123,456)")
    seed_group.add_argument(
        "-n",
        "--num-variants",
        "--num_variants",
        dest="num_variants",
        type=int,
        default=None,
        help="Number of random variants per input (e.g. -n 3)",
    )
    parser.add_argument(
        "--source-image",
        "--source_image",
        dest="source_image",
        type=str,
        default=None,
        help="Source image for video synthesis (used with --vision-npy)",
    )
    parser.add_argument(
        "--save-muq", "--save_muq", dest="save_muq", action="store_true", help="Also save MuQ embeddings"
    )
    parser.add_argument(
        "--no-video",
        "--no_video",
        dest="no_video",
        action="store_true",
        help="Skip static image and audio MP4 synthesis",
    )

    args = parser.parse_args()

    if (args.video is not None or args.text is not None) and args.backbone != "qwen3vl":
        parser.error("--video and --text require --backbone qwen3vl")
    if args.video_max_frames < 1:
        parser.error("--video-max-frames must be at least 1")
    if args.rdm_steps < 1 or args.sample_steps < 1 or args.rdm_ensemble < 1:
        parser.error("--rdm-steps, --sample-steps, and --rdm-ensemble must be at least 1")
    if not math.isfinite(args.rdm_guidance) or not math.isfinite(args.guidance_scale):
        parser.error("guidance scales must be finite")
    if not math.isfinite(args.rdm_scale_factor) or args.rdm_scale_factor <= 0:
        parser.error("--rdm-scale-factor must be positive and finite")
    if args.num_variants is not None and args.num_variants < 1:
        parser.error("--num-variants must be at least 1")
    if args.source_image is not None and args.vision_npy is None:
        parser.error("--source-image can only be used with --vision-npy")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("a CUDA device was requested, but CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error(f"CUDA device {device.index} is unavailable; detected {torch.cuda.device_count()} device(s)")

    # Resolve seeds: explicit list, random N, or default single
    import random

    if args.seeds is not None:
        try:
            seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        except ValueError:
            parser.error("--seeds must be a comma-separated list of integers")
        if not seeds:
            parser.error("--seeds must contain at least one integer")
    elif args.num_variants is not None:
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_variants)]
    else:
        seeds = [42]
    invalid_seeds = [seed for seed in seeds if not 0 <= seed <= _MAX_SEED]
    if invalid_seeds:
        parser.error(f"seeds must be between 0 and {_MAX_SEED}; got {invalid_seeds[0]}")
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds must not contain duplicates")

    # Validate all local input and explicit checkpoint paths before downloads.
    context_dims = {"clip": 1024, "qwen3vl": 2048}
    stage1_input = any(value is not None for value in (args.image, args.video, args.text, args.vision_npy))
    images = None
    video_path = None
    vision_condition = None
    muq_data = None

    if args.text is not None and not args.text.strip():
        parser.error("--text must be non-empty")
    if args.image is not None:
        try:
            images = collect_images(Path(args.image).expanduser())
        except FileNotFoundError as error:
            parser.error(str(error))
        if not images:
            parser.error("--image directory contains no supported image files")
    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            parser.error(f"video file not found: {video_path}")
    if args.muq_npy is not None:
        try:
            muq_data = _load_embedding_file(args.muq_npy, 512, "MuQ")
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    if args.vision_npy is not None:
        try:
            vision_condition = _load_embedding_file(
                args.vision_npy,
                context_dims[args.backbone],
                f"{args.backbone} condition",
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    if args.source_image is not None:
        source_image = Path(args.source_image).expanduser().resolve()
        if not source_image.is_file():
            parser.error(f"source image not found: {source_image}")
    else:
        source_image = None

    output_root = Path(args.output_dir).expanduser()
    if output_root.exists() and not output_root.is_dir():
        parser.error(f"--output-dir is not a directory: {output_root}")
    for value, label in ((args.rdm_path, "--rdm-path"), (args.stage2_ckpt, "--stage2-ckpt")):
        if value is not None and not Path(value).expanduser().is_file():
            parser.error(f"{label} is not a file: {Path(value).expanduser()}")

    # Resolve project checkpoints only after input validation.
    if args.rdm_path is None and stage1_input:
        if args.backbone == "clip":
            parser.error("--rdm-path is required for the bring-your-own CLIP backbone")
        from meric import hub

        args.rdm_path = str(hub.resolve_rdm("meric-sft-v3"))
        print(f"Resolved Music Head: {args.rdm_path}")

    if args.stage2_ckpt is None:
        from meric import hub

        args.stage2_ckpt = str(hub.resolve_stage2())

    # Create a collision-resistant run directory after successful resolution.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 70)
    print("Meric Inference Pipeline")
    print("=" * 70)
    if args.image:
        print("  Mode:       Image -> Music")
        print(f"  Backbone:   {args.backbone} (dim={context_dims[args.backbone]})")
        print(f"  RDM:        {args.rdm_path}")
    elif args.video:
        print("  Mode:       Video -> Qwen3-VL -> Music Head -> Music")
        print(f"  Video:      {args.video}")
        print(f"  RDM:        {args.rdm_path}")
    elif args.vision_npy:
        print("  Mode:       Vision embedding (.npy) -> RDM -> Music")
        print(f"  Backbone:   {args.backbone} (dim={context_dims[args.backbone]})")
        print(f"  Vision npy: {args.vision_npy}")
        print(f"  RDM:        {args.rdm_path}")
    elif args.text:
        print("  Mode:       Text -> Qwen3-VL -> Music Head -> Music")
        print(f"  Prompt:     {args.text}")
    elif args.muq_npy:
        print("  Mode:       MuQ embedding -> Music")
        print(f"  MuQ file:   {args.muq_npy}")
    print(f"  Stage 2:    steps={args.sample_steps}, cfg={args.guidance_scale}, method={args.sample_method}")
    print(f"  Seeds:      {seeds}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    # ==========================================
    # Prepare MuQ embeddings based on input mode
    # ==========================================
    # muq_items: list of (name, muq_tensor [1,512], source_image_path, Stage-2 seed)
    muq_items = []

    if args.muq_npy:
        name = _safe_name(Path(args.muq_npy).stem, "muq")
        for seed in seeds:
            suffix = f"_seed{seed}" if len(seeds) > 1 else ""
            muq_items.append((f"{name}{suffix}", torch.tensor(muq_data, dtype=torch.float32), None, seed))
    else:
        conditions = []
        input_dir = output_dir / "input"

        if args.text:
            condition = _extract_qwen3vl_text_via_subprocess([args.text], output_dir, args.device)
            name = _safe_name(args.text, "text")
            conditions.append((name, condition, None))
        elif args.video:
            input_dir.mkdir(exist_ok=True)
            shutil.copy2(video_path, input_dir / video_path.name)
            video_embeddings = _extract_qwen3vl_video_via_subprocess(
                [video_path], output_dir, args.device, max_frames=args.video_max_frames
            )
            conditions.append((_safe_name(video_path.stem, "video"), video_embeddings[str(video_path)], None))
        elif args.vision_npy:
            src_img = None
            if source_image is not None:
                input_dir.mkdir(exist_ok=True)
                shutil.copy2(source_image, input_dir / source_image.name)
                src_img = input_dir / source_image.name
            conditions.append((_safe_name(Path(args.vision_npy).stem, "condition"), vision_condition, src_img))
        else:
            input_dir.mkdir(exist_ok=True)
            for image_path in images:
                shutil.copy2(image_path, input_dir / image_path.name)

            if args.backbone == "clip":
                encoder, preprocess = load_clip_encoder(device)
                embeddings = {
                    str(image_path.resolve()): extract_clip_embedding(encoder, preprocess, image_path, device)
                    for image_path in images
                }
                del encoder
                torch.cuda.empty_cache()
            else:
                embeddings = _extract_qwen3vl_via_subprocess(images, output_dir, args.device)

            for image_path in images:
                conditions.append(
                    (
                        _safe_name(image_path.stem, "image"),
                        embeddings[str(image_path.resolve())],
                        input_dir / image_path.name,
                    )
                )

        expected_dim = context_dims[args.backbone]
        for name, condition, _ in conditions:
            if condition.shape != (1, expected_dim):
                raise ValueError(f"Condition {name!r} has shape {condition.shape}; expected (1, {expected_dim})")
            if not np.isfinite(condition).all():
                raise ValueError(f"Condition {name!r} contains non-finite values")

        rdm_model, rdm_scheduler = load_rdm(args.rdm_path, expected_dim, device)
        for base_name, condition, source_image in conditions:
            if args.rdm_ensemble > 1:
                ensemble = [
                    rdm_generate_muq(
                        rdm_model,
                        rdm_scheduler,
                        condition,
                        device,
                        num_steps=args.rdm_steps,
                        guidance_scale=args.rdm_guidance,
                        scale_factor=args.rdm_scale_factor,
                        seed=ensemble_seed,
                    )
                    for ensemble_seed in range(args.rdm_ensemble)
                ]
                shared_muq = torch.stack(ensemble).mean(dim=0)
            else:
                shared_muq = None

            for seed in seeds:
                muq = shared_muq
                if muq is None:
                    muq = rdm_generate_muq(
                        rdm_model,
                        rdm_scheduler,
                        condition,
                        device,
                        num_steps=args.rdm_steps,
                        guidance_scale=args.rdm_guidance,
                        scale_factor=args.rdm_scale_factor,
                        seed=seed,
                    )
                suffix = f"_seed{seed}" if len(seeds) > 1 else ""
                item_name = f"{base_name}{suffix}"
                muq_items.append((item_name, muq, source_image, seed))
                if args.save_muq:
                    np.save(output_dir / f"{item_name}_muq.npy", muq.numpy())

        del rdm_model
        torch.cuda.empty_cache()

    if not muq_items:
        print("Nothing to generate.")
        return

    print(f"\nTotal samples to generate: {len(muq_items)}")

    # ==========================================
    # Stage 2: MuQ → Audio
    # ==========================================
    from meric import hub

    vae_dir, transformer_dir = hub.resolve_stable_audio()
    muq_dir = hub.resolve_muq_mulan()
    meric_model = load_meric_ldm(
        device,
        args.stage2_ckpt,
        guidance_scale=args.guidance_scale,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
        vae_dir=vae_dir,
        transformer_dir=transformer_dir,
        muq_dir=muq_dir,
    )

    results = []
    for i, (name, muq_emb, src_img, generation_seed) in enumerate(muq_items):
        print(f"\n[{i + 1}/{len(muq_items)}] Generating audio for: {name}")
        t0 = time.time()
        wav_path = generate_audio(meric_model, muq_emb, output_dir, name, device, seed=generation_seed)
        elapsed = time.time() - t0
        print(f"  Saved: {wav_path} ({elapsed:.1f}s)")
        results.append(
            {
                "name": name,
                "wav": str(wav_path),
                "source_image": str(src_img) if src_img else None,
                "seed": generation_seed,
                "time": elapsed,
                "sha256": _sha256(wav_path),
            }
        )

    # ==========================================
    # Video Synthesis: image + audio → MP4
    # ==========================================
    if not args.no_video:
        print(f"\n{'=' * 70}")
        print("Synthesizing videos (image + audio -> MP4)")
        print("=" * 70)

        video_dir = output_dir / "video"
        video_dir.mkdir(exist_ok=True)

        # Group results by source image for multi-seed concat
        # image_stem -> list of (name, wav_path, video_path)
        from collections import defaultdict

        image_groups = defaultdict(list)

        for r in results:
            src_img = r["source_image"]
            if src_img is None:
                continue
            wav = Path(r["wav"])
            if not wav.exists():
                continue

            video_path = video_dir / f"{r['name']}.mp4"
            vp = make_video(src_img, wav, video_path)
            if vp:
                print(f"  {video_path.name}")
                r["video"] = str(video_path)
                # Group by image stem (without _seedN suffix)
                img_stem = Path(src_img).stem
                image_groups[img_stem].append(video_path)

        # Create concatenated comparison videos for multi-seed cases
        if len(seeds) > 1:
            for img_stem, video_paths in image_groups.items():
                if len(video_paths) < 2:
                    continue
                concat_path = video_dir / f"{img_stem}_all_seeds.mp4"
                cp = make_concat_video(sorted(video_paths), concat_path)
                if cp:
                    print(f"  {concat_path.name} (concat {len(video_paths)} seeds)")

    # Save manifest
    manifest = {
        "timestamp": timestamp,
        "input_mode": (
            "image"
            if args.image
            else "video"
            if args.video
            else "text"
            if args.text
            else "vision_npy"
            if args.vision_npy
            else "muq"
        ),
        "input_source": args.image or args.video or args.text or args.vision_npy or args.muq_npy,
        "backbone": args.backbone if stage1_input else None,
        "rdm_path": args.rdm_path,
        "stage2_ckpt": args.stage2_ckpt,
        "guidance_scale": args.guidance_scale,
        "sample_steps": args.sample_steps,
        "sample_method": args.sample_method,
        "rdm_steps": args.rdm_steps if stage1_input else None,
        "rdm_guidance": args.rdm_guidance if stage1_input else None,
        "rdm_ensemble": args.rdm_ensemble if stage1_input else None,
        "seeds": seeds,
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Done! Generated {len(results)} audio file(s)")
    if not args.no_video and args.image:
        video_count = sum(1 for r in results if "video" in r)
        print(
            f"  Videos:     {video_count} individual"
            + (f" + {len([g for g in image_groups.values() if len(g) > 1])} concatenated" if len(seeds) > 1 else "")
        )
    print(f"  Output dir: {output_dir}")
    print(f"  Manifest:   {manifest_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
