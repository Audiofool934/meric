"""High-level Meric inference facade.

    from meric import MericPipeline
    pipe = MericPipeline.from_pretrained("meric-sft-v3", device="cuda:0")
    wavs = pipe.generate(image="photo.jpg", n=3, output_dir="outputs/")

Wraps the staged two-stage engine in :mod:`meric.inference.run_pipeline`
(Stage 1: image/video/text -> MuQ via the Music Head; Stage 2: MuQ -> audio via
MericLDM), loading every model once and resolving weights through
:mod:`meric.hub`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from meric import __version__, hub
from meric.inference import run_pipeline as _engine

_MAX_SEED = 2**63 - 1


def _normalize_seeds(seed: int, n: int, seeds) -> list[int]:
    values = list(seeds) if seeds is not None else [seed + index for index in range(n)]
    if not values:
        raise ValueError("seeds must not be empty")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Every seed must be an integer, got {value!r}")
        value = int(value)
        if not 0 <= value <= _MAX_SEED:
            raise ValueError(f"Every seed must be between 0 and {_MAX_SEED}, got {value}")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("seeds must not contain duplicates")
    return normalized


def _coerce_muq_array(value) -> np.ndarray:
    """Convert one in-memory MuQ embedding to finite float32 shape ``[1, 512]``."""
    try:
        raw = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("muq must be a numeric array, tensor, or .npy path") from error
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("muq must contain real numeric values")
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape != (1, 512):
        raise ValueError(f"Expected one MuQ embedding with shape (512,) or (1, 512), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("MuQ embedding contains non-finite values")
    return array


def _validate_generation_request(*, image, video, text, muq, seed, n, seeds, video_max_frames) -> list[int]:
    provided = [value for value in (image, video, text, muq) if value is not None]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of image=, video=, text=, or muq=")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer (at least 1)")
    if isinstance(video_max_frames, bool) or not isinstance(video_max_frames, Integral) or video_max_frames < 1:
        raise ValueError("video_max_frames must be a positive integer")
    normalized_seeds = _normalize_seeds(seed, int(n), seeds)

    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("text must be a non-empty string")
    for value, label in ((image, "Image"), (video, "Video")):
        if value is not None:
            try:
                path = Path(value).expanduser()
            except TypeError as error:
                raise TypeError(f"{label.lower()} must be a filesystem path") from error
            if not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path.resolve()}")
    if isinstance(muq, str | os.PathLike):
        _engine._load_embedding_file(muq, 512, "MuQ")
    elif muq is not None:
        _coerce_muq_array(muq)

    return normalized_seeds


def _validate_runtime_config(
    *,
    backbone,
    context_dim,
    device,
    rdm_steps,
    rdm_guidance,
    rdm_scale_factor,
    guidance_scale,
    sample_steps,
    sample_method,
) -> torch.device:
    if backbone not in {"clip", "qwen3vl"}:
        raise ValueError("backbone must be 'clip' or 'qwen3vl'")
    if not isinstance(context_dim, int) or context_dim < 1:
        raise ValueError("context_dim must be a positive integer")
    if not isinstance(rdm_steps, int) or rdm_steps < 1:
        raise ValueError("rdm_steps must be a positive integer")
    if not isinstance(sample_steps, int) or sample_steps < 1:
        raise ValueError("sample_steps must be a positive integer")
    if not math.isfinite(rdm_guidance) or not math.isfinite(guidance_scale):
        raise ValueError("guidance scales must be finite")
    if not math.isfinite(rdm_scale_factor) or rdm_scale_factor <= 0:
        raise ValueError("rdm_scale_factor must be positive and finite")
    if sample_method not in {"euler", "midpoint", "dopri5"}:
        raise ValueError("sample_method must be one of: euler, midpoint, dopri5")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available")
        if resolved_device.index is not None and resolved_device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {resolved_device.index} is unavailable; "
                f"detected {torch.cuda.device_count()} device(s)"
            )
    return resolved_device


class MericPipeline:
    """A loaded Meric model for image, video, text, or MuQ-conditioned audio."""

    def __init__(
        self,
        rdm_model,
        rdm_sampler,
        meric_ldm,
        *,
        backbone,
        context_dim,
        device,
        model_name="custom",
        rdm_steps=20,
        rdm_guidance=2.0,
        rdm_scale_factor=10.0,
        guidance_scale=5.0,
        sample_steps=50,
        sample_method="dopri5",
    ):
        resolved_device = _validate_runtime_config(
            backbone=backbone,
            context_dim=context_dim,
            device=device,
            rdm_steps=rdm_steps,
            rdm_guidance=rdm_guidance,
            rdm_scale_factor=rdm_scale_factor,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            sample_method=sample_method,
        )
        self.rdm_model = rdm_model
        self.rdm_sampler = rdm_sampler
        self.meric_ldm = meric_ldm
        self.backbone = backbone
        self.context_dim = context_dim
        self.device = resolved_device
        self.model_name = model_name
        self.rdm_steps = rdm_steps
        self.rdm_guidance = rdm_guidance
        self.rdm_scale_factor = rdm_scale_factor
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        # CLIP is retained for bring-your-own Stage-1 checkpoints.
        self._clip = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "meric-sft-v3",
        *,
        device: str = "cuda:0",
        rdm_steps: int = 20,
        rdm_guidance: float = 2.0,
        rdm_scale_factor: float = 10.0,
        guidance_scale: float = 5.0,
        sample_steps: int = 50,
        sample_method: str = "dopri5",
    ) -> MericPipeline:
        """Load a pretrained Meric model by name (see ``meric.list_models()``)."""
        if model_name not in hub.MODELS:
            raise ValueError(f"Unknown model {model_name!r}. Available models: {', '.join(hub.MODELS)}")
        spec = hub.MODELS[model_name]
        _validate_runtime_config(
            backbone=spec["backbone"],
            context_dim=spec["context_dim"],
            device=device,
            rdm_steps=rdm_steps,
            rdm_guidance=rdm_guidance,
            rdm_scale_factor=rdm_scale_factor,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            sample_method=sample_method,
        )
        rdm_path = hub.resolve_rdm(model_name)
        stage2_ckpt = hub.resolve_stage2()
        vae_dir, transformer_dir = hub.resolve_stable_audio()
        muq_dir = hub.resolve_muq_mulan()

        rdm_model, rdm_sampler = _engine.load_rdm(str(rdm_path), spec["context_dim"], device)
        meric_ldm = _engine.load_meric_ldm(
            device,
            str(stage2_ckpt),
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            sample_method=sample_method,
            vae_dir=vae_dir,
            transformer_dir=transformer_dir,
            muq_dir=muq_dir,
        )
        return cls(
            rdm_model,
            rdm_sampler,
            meric_ldm,
            backbone=spec["backbone"],
            context_dim=spec["context_dim"],
            device=device,
            model_name=model_name,
            rdm_steps=rdm_steps,
            rdm_guidance=rdm_guidance,
            rdm_scale_factor=rdm_scale_factor,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            sample_method=sample_method,
        )

    # ------------------------------------------------------------------ #
    def _ensure_clip(self):
        if self._clip is None:
            self._clip = _engine.load_clip_encoder(self.device)
        return self._clip

    def _vision_embedding(self, image_path: Path, output_dir: Path) -> np.ndarray:
        if self.backbone == "clip":
            model, preprocess = self._ensure_clip()
            return _engine.extract_clip_embedding(model, preprocess, image_path, self.device)
        # Qwen3-VL has an isolated environment because its dependency versions
        # conflict with the Stage-2 runtime.
        embs = _engine._extract_qwen3vl_via_subprocess([image_path], output_dir, str(self.device))
        return embs[str(image_path.resolve())]

    def _text_embedding(self, text: str, output_dir: Path) -> np.ndarray:
        if self.backbone != "qwen3vl":
            raise ValueError("Text generation requires a Qwen3-VL Music Head checkpoint")
        return _engine._extract_qwen3vl_text_via_subprocess([text], output_dir, str(self.device))

    def _video_embedding(self, video_path: Path, output_dir: Path, max_frames: int) -> np.ndarray:
        if self.backbone != "qwen3vl":
            raise ValueError("Video generation requires a Qwen3-VL Music Head checkpoint")
        embs = _engine._extract_qwen3vl_video_via_subprocess(
            [video_path], output_dir, str(self.device), max_frames=max_frames
        )
        return embs[str(video_path.resolve())]

    def _muq_from_condition(self, condition: np.ndarray, seeds: list[int]) -> list[torch.Tensor]:
        condition = np.asarray(condition, dtype=np.float32)
        if condition.shape != (1, self.context_dim):
            raise ValueError(f"Expected a (1, {self.context_dim}) Stage-1 condition, got {condition.shape}")
        if not np.isfinite(condition).all():
            raise ValueError("Stage-1 condition contains non-finite values")
        out = []
        for seed in seeds:
            muq = _engine.rdm_generate_muq(
                self.rdm_model,
                self.rdm_sampler,
                condition,
                self.device,
                num_steps=self.rdm_steps,
                guidance_scale=self.rdm_guidance,
                scale_factor=self.rdm_scale_factor,
                seed=seed,
            )
            if muq.shape != (1, 512) or not torch.isfinite(muq).all():
                raise RuntimeError(f"Music Head returned an invalid MuQ tensor with shape {tuple(muq.shape)}")
            out.append(muq)
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        *,
        image=None,
        video=None,
        text=None,
        muq=None,
        seed: int = 42,
        n: int = 1,
        seeds=None,
        output_dir=None,
        video_max_frames: int = 8,
    ):
        """Generate audio from exactly one supported input modality.

        - ``image``: path to an image file (uses the configured vision backbone).
        - ``video``: path to a video file, sampled to at most ``video_max_frames``.
        - ``text``: text prompt encoded by Qwen3-VL and the Music Head.
        - ``muq``: one pre-computed MuQ embedding, which skips Stage 1.

        ``seed`` controls both Music Head sampling and Stage-2 synthesis. Pass
        ``seeds`` to choose exact variants; otherwise ``n`` consecutive seeds
        beginning at ``seed`` are used.

        Returns a list of paths to the generated WAV files.
        """
        seeds = _validate_generation_request(
            image=image,
            video=video,
            text=text,
            muq=muq,
            seed=seed,
            n=n,
            seeds=seeds,
            video_max_frames=video_max_frames,
        )
        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="meric_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        source_path = None

        def safe_name(value: str, fallback: str) -> str:
            name = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
            return name[:64] or fallback

        def checked_path(value, label: str) -> Path:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path}")
            return path

        # Build (name, MuQ tensor, Stage-2 seed) items.
        items = []
        if image is not None:
            img = checked_path(image, "Image")
            source_path = img
            condition = self._vision_embedding(img, out_dir)
            muqs = self._muq_from_condition(condition, seeds)
            for s, m in zip(seeds, muqs, strict=True):
                suffix = f"_seed{s}" if len(seeds) > 1 else ""
                items.append((f"{safe_name(img.stem, 'image')}{suffix}", m, s))
        elif video is not None:
            video_path = checked_path(video, "Video")
            source_path = video_path
            condition = self._video_embedding(video_path, out_dir, video_max_frames)
            muqs = self._muq_from_condition(condition, seeds)
            for s, m in zip(seeds, muqs, strict=True):
                suffix = f"_seed{s}" if len(seeds) > 1 else ""
                items.append((f"{safe_name(video_path.stem, 'video')}{suffix}", m, s))
        elif text is not None:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            condition = self._text_embedding(text, out_dir)
            muqs = self._muq_from_condition(condition, seeds)
            base_name = safe_name(text, "text")
            for s, m in zip(seeds, muqs, strict=True):
                suffix = f"_seed{s}" if len(seeds) > 1 else ""
                items.append((f"{base_name}{suffix}", m, s))
        else:  # muq
            if isinstance(muq, str | os.PathLike):
                muq_path = checked_path(muq, "MuQ")
                source_path = muq_path
                arr = _engine._load_embedding_file(muq_path, 512, "MuQ")
                base_name = safe_name(muq_path.stem, "muq")
            else:
                arr = _coerce_muq_array(muq)
                base_name = "muq"
            if arr.shape != (1, 512):
                raise ValueError(f"Expected one MuQ embedding with shape (512,) or (1, 512), got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError("MuQ embedding contains non-finite values")
            embedding = torch.as_tensor(np.asarray(arr, dtype=np.float32))
            for s in seeds:
                suffix = f"_seed{s}" if len(seeds) > 1 else ""
                items.append((f"{base_name}{suffix}", embedding, s))

        # Stage 2: MuQ -> audio.
        paths = []
        for name, muq_emb, generation_seed in items:
            wav = _engine.generate_audio(
                self.meric_ldm,
                muq_emb,
                out_dir,
                name,
                self.device,
                seed=generation_seed,
            )
            path = Path(wav)
            if not path.is_file():
                raise RuntimeError(f"Stage 2 did not create the expected audio file: {path}")
            paths.append(path)

        input_kind = next(
            kind
            for kind, value in (("image", image), ("video", video), ("text", text), ("muq", muq))
            if value is not None
        )
        input_value = {"type": input_kind}
        if input_kind == "text":
            input_value["text"] = text
        elif source_path is not None:
            input_value["path"] = str(source_path)
            input_value["sha256"] = _sha256(source_path)
        else:
            input_array = np.ascontiguousarray(embedding.detach().cpu().numpy(), dtype=np.float32)
            input_value.update(
                {
                    "source": "in-memory",
                    "shape": list(input_array.shape),
                    "dtype": str(input_array.dtype),
                    "sha256": hashlib.sha256(input_array.tobytes()).hexdigest(),
                }
            )

        outputs = []
        for generation_seed, path in zip(seeds, paths, strict=True):
            item = {"seed": generation_seed, "path": str(path)}
            if path.is_file():
                item["sha256"] = _sha256(path)
            outputs.append(item)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "meric": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(self.device),
            },
            "model": {
                "name": getattr(self, "model_name", "custom"),
                "meric_configured_revision": hub._MERIC_REVISION,
                "music_head_expected_sha256": hub.MODELS.get(getattr(self, "model_name", ""), {}).get("sha256"),
                "stage2_expected_sha256": hub.STAGE2["sha256"],
                "stable_audio_configured_revision": hub.STABLE_AUDIO_REVISION,
                "muq_mulan_configured_revision": hub.MUQ_MULAN_REVISION,
                "qwen3vl_code_tested_revision": hub.QWEN3VL_CODE_REVISION
                if input_kind != "muq" and self.backbone == "qwen3vl"
                else None,
                "qwen3vl_model_tested_revision": hub.QWEN3VL_MODEL_REVISION
                if input_kind != "muq" and self.backbone == "qwen3vl"
                else None,
            },
            "input": input_value,
            "settings": {
                "rdm_steps": self.rdm_steps if input_kind != "muq" else None,
                "rdm_guidance": self.rdm_guidance if input_kind != "muq" else None,
                "rdm_scale_factor": self.rdm_scale_factor if input_kind != "muq" else None,
                "guidance_scale": getattr(self, "guidance_scale", None),
                "sample_steps": getattr(self, "sample_steps", None),
                "sample_method": getattr(self, "sample_method", None),
                "audio_sample_rate": getattr(self.meric_ldm, "audio_sample_rate", None),
                "video_max_frames": video_max_frames if input_kind == "video" else None,
            },
            "outputs": outputs,
        }
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    *,
    image=None,
    video=None,
    text=None,
    muq=None,
    model_name: str = "meric-sft-v3",
    device: str = "cuda:0",
    output_dir=None,
    n: int = 1,
    seed: int = 42,
    seeds=None,
    video_max_frames: int = 8,
    **model_kwargs,
):
    """One-shot convenience wrapper: load ``model_name`` and generate in a single call."""
    _validate_generation_request(
        image=image,
        video=video,
        text=text,
        muq=muq,
        seed=seed,
        n=n,
        seeds=seeds,
        video_max_frames=video_max_frames,
    )
    pipe = MericPipeline.from_pretrained(model_name, device=device, **model_kwargs)
    return pipe.generate(
        image=image,
        video=video,
        text=text,
        muq=muq,
        n=n,
        seed=seed,
        seeds=seeds,
        output_dir=output_dir,
        video_max_frames=video_max_frames,
    )
