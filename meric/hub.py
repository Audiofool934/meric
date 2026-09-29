"""Checkpoint resolution for Meric.

Weights are distributed via the Hugging Face Hub and cached locally - no weights
are tracked in git. Each model is referenced by a short name (see ``MODELS``);
``MericPipeline.from_pretrained(name)`` resolves and downloads on first use.

Resolution order for every checkpoint:
  1. An explicit path / environment override (e.g. ``MERIC_STAGE2_CKPT``).
  2. A local file under ``$MERIC_HOME`` (default ``~/.cache/meric``) - lets you
     point at already-downloaded weights without any network access.
  3. ``huggingface_hub`` download from the registered repo.

Stage-2 audio components (Stable Audio Open VAE/DiT and MuQ-MuLan) are public
upstream models fetched directly from their canonical Hugging Face repos.
"""

import os
from pathlib import Path

_DEFAULT_MERIC_REPO = "Audiofool/meric"
_DEFAULT_MERIC_REVISION = "0a199f0f6a0c7f64f397a2192882883bfb2c7736"


def _repo_and_revision(repo_env: str, revision_env: str, default_repo: str, default_revision: str):
    """Resolve a repository mirror without applying an unrelated default pin."""
    repo = os.environ.get(repo_env, default_repo)
    revision = os.environ.get(revision_env)
    if revision is None and repo == default_repo:
        revision = default_revision
    return repo, revision


# Public Meric model repository. Mirrors can opt out of the release pin by
# overriding MERIC_HF_REPO without setting MERIC_HF_REVISION.
_MERIC_REPO, _MERIC_REVISION = _repo_and_revision(
    "MERIC_HF_REPO", "MERIC_HF_REVISION", _DEFAULT_MERIC_REPO, _DEFAULT_MERIC_REVISION
)

#: Stage-1 "Music Head" models: short name -> resolution spec.
#: Both are Qwen3-VL-backbone heads that share the one Stage-2 Flow Decoder (STAGE2).
MODELS = {
    "meric-sft-v3": dict(
        repo_id=_MERIC_REPO,
        revision=_MERIC_REVISION,
        filename="rdm_sft_v3.pth",
        backbone="qwen3vl",
        context_dim=2048,
        bytes=2_025_788_845,
        sha256="1f53e8d7bdae7309f48a4cecea2f0a3b92166ba8f09de37ad70a57ccbe595320",
        desc="Camera-ready Music Head: ARIA fine-tuning with a squared-cosine schedule, v-prediction, EMA, and min-SNR.",
    ),
    "meric-instrumental": dict(
        repo_id=_MERIC_REPO,
        revision=_MERIC_REVISION,
        filename="rdm_sft_instrumental.pth",
        backbone="qwen3vl",
        context_dim=2048,
        bytes=2_025_788_845,
        sha256="09b290a6f31dccc3d0b521eb61b372bb368288d22203b592f0d6c62ca2d3b2c3",
        desc="Non-vocal head: vocal-filtered ARIA fine-tune for cleaner instrumental output.",
    ),
}

#: Stage-2 MericLDM (MuQ -> audio) checkpoint.
STAGE2 = dict(
    repo_id=_MERIC_REPO,
    revision=_MERIC_REVISION,
    filename="mericldm.ckpt",
    bytes=15_903_442_922,
    sha256="f43d8abc8d7a7aabe60ef68f29bce544641d236681f44f9ecadd33607fa4d47a",
)

#: Upstream public component repos used by Stage-2.
_DEFAULT_STABLE_AUDIO_REPO = "stabilityai/stable-audio-open-1.0"
_DEFAULT_STABLE_AUDIO_REVISION = "f21265c1e2710b3bd2386596943f0007f55f802e"
STABLE_AUDIO_REPO, STABLE_AUDIO_REVISION = _repo_and_revision(
    "MERIC_STABLE_AUDIO_REPO",
    "MERIC_STABLE_AUDIO_REVISION",
    _DEFAULT_STABLE_AUDIO_REPO,
    _DEFAULT_STABLE_AUDIO_REVISION,
)
_DEFAULT_MUQ_MULAN_REPO = "OpenMuQ/MuQ-MuLan-large"
_DEFAULT_MUQ_MULAN_REVISION = "2e01c796b71dca71b45251384c04cd7b237c9020"
MUQ_MULAN_REPO, MUQ_MULAN_REVISION = _repo_and_revision(
    "MERIC_MUQ_REPO", "MERIC_MUQ_REVISION", _DEFAULT_MUQ_MULAN_REPO, _DEFAULT_MUQ_MULAN_REVISION
)
QWEN3VL_CODE_REPO = "https://github.com/QwenLM/Qwen3-VL-Embedding.git"
QWEN3VL_CODE_REVISION = "8ce3aab1fbcc7b7143b094f0b2005dd89e7246b9"
QWEN3VL_MODEL_REPO = "Qwen/Qwen3-VL-Embedding-2B"
QWEN3VL_MODEL_REVISION = "2a50926d213628c727f38025982a76f655673f54"


def meric_home() -> Path:
    return Path(os.environ.get("MERIC_HOME", Path.home() / ".cache" / "meric"))


def list_models():
    """Return the available pretrained model names and descriptions."""
    return {k: v["desc"] for k, v in MODELS.items()}


def _resolve(repo_id, filename, env_override=None, revision=None):
    # 1) explicit env override
    if env_override:
        p = os.environ.get(env_override)
        if p:
            path = Path(p).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{env_override} is not a checkpoint file: {path}")
            return path
    # 2) local cache under MERIC_HOME
    local = meric_home() / filename
    if local.is_file():
        return local.resolve()
    if local.exists():
        raise FileNotFoundError(f"Expected a checkpoint file, found a non-file path: {local}")
    # 3) HF Hub download
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)).resolve()
    if not downloaded.is_file():
        raise FileNotFoundError(f"Hugging Face download did not return a checkpoint file: {downloaded}")
    return downloaded


def resolve_rdm(model_name: str) -> Path:
    """Resolve the Stage-1 Music Head checkpoint for ``model_name``."""
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")
    spec = MODELS[model_name]
    return _resolve(
        spec["repo_id"],
        spec["filename"],
        env_override="MERIC_RDM_CKPT",
        revision=spec["revision"],
    )


def resolve_stage2() -> Path:
    """Resolve the Stage-2 MericLDM checkpoint."""
    return _resolve(
        STAGE2["repo_id"],
        STAGE2["filename"],
        env_override="MERIC_STAGE2_CKPT",
        revision=STAGE2["revision"],
    )


def resolve_stable_audio():
    """Return (vae_dir, transformer_dir) for Stable Audio Open, downloading if needed."""
    root = os.environ.get("MERIC_STABLE_AUDIO_DIR")
    if not root:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_id=STABLE_AUDIO_REPO,
            revision=STABLE_AUDIO_REVISION,
            allow_patterns=["vae/*", "transformer/*", "*.json"],
        )
    root = Path(root).expanduser().resolve()
    vae = root / "vae"
    transformer = root / "transformer"
    if not vae.is_dir() or not transformer.is_dir():
        raise FileNotFoundError(f"Stable Audio root must contain vae/ and transformer/: {root}")
    return str(vae), str(transformer)


def resolve_muq_mulan() -> str:
    """Return the pinned MuQ-MuLan model directory, downloading if needed."""
    root = os.environ.get("MERIC_MUQ_DIR")
    if not root:
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id=MUQ_MULAN_REPO, revision=MUQ_MULAN_REVISION)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MuQ-MuLan model directory not found: {root}")
    return str(root)
