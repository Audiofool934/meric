# Environment setup

This guide installs the supported Meric inference and training environment.
The tested platform is Linux with Python 3.10, PyTorch 2.5.1, and CUDA 12.1.

## Requirements

- Linux on x86-64
- An NVIDIA GPU and CUDA-compatible driver
- Python 3.10
- Git
- Approximately 30 GB of free disk space for all default weights
- `ffmpeg` only if you use the low-level static-image video renderer

PyTorch Lightning is pinned to 1.9.3 because the training modules use its pre-2.0 interfaces.
The runtime also pins `setuptools<81` because Lightning 1.9 imports `pkg_resources`.

## Install with venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

Install development tools only when needed:

```bash
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the authoritative package metadata.
`requirements.txt` mirrors the pinned core runtime for workflows that require a requirements file.

## Install with Conda

```bash
conda env create -f environment.yml
conda activate meric

python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

## Qwen3-VL environment

The released Music Heads use Qwen3-VL-Embedding-2B for image, video, and text inputs.
Qwen3-VL currently requires Python 3.11 and PyTorch 2.8, so Meric runs it in a separate environment through a JSON subprocess protocol.

Install the tested [`uv`](https://docs.astral.sh/uv/getting-started/installation/) version and run:

```bash
python -m pip install uv==0.9.26
bash scripts/setup_qwen3vl.sh
export QWEN3VL_REPO="$PWD/.external/Qwen3-VL-Embedding"
```

The helper performs four reproducible operations:

1. It clones `https://github.com/QwenLM/Qwen3-VL-Embedding.git`.
2. It checks out `8ce3aab1fbcc7b7143b094f0b2005dd89e7246b9`.
3. It verifies the upstream `uv.lock`, exports its exact hash-locked dependency graph, and installs it from `https://pypi.org/simple` into an isolated Python 3.11 environment.
4. Unless `QWEN3VL_MODEL` already points to a local directory, it downloads `Qwen/Qwen3-VL-Embedding-2B` at revision `2a50926d213628c727f38025982a76f655673f54`.

The pinned upstream project declares an insecure HTTP package mirror.
The helper intentionally does not run the upstream sync command and does not contact that mirror.

To use an existing checkout, set `QWEN3VL_REPO` before running the helper.
To use an existing model directory, export `QWEN3VL_MODEL=/absolute/path/to/Qwen3-VL-Embedding-2B`.
Meric warns when a Git checkout is not at the tested code revision.

## Environment variables

Meric reads environment variables directly.
It does not automatically parse a `.env` file.
If you use `.env.example` as a template, export the variables in your shell or source the file explicitly.

| Variable | Purpose |
|---|---|
| `QWEN3VL_REPO` | External Qwen3-VL-Embedding checkout with `.venv/` |
| `QWEN3VL_MODEL` | Optional override for the Qwen3-VL-Embedding-2B model directory |
| `MERIC_HOME` | Local Meric checkpoint directory, default `~/.cache/meric` |
| `MERIC_HF_REPO` | Override the default model repository `Audiofool/meric` |
| `MERIC_HF_REVISION` | Override the Meric Hub revision |
| `MERIC_RDM_CKPT` | Local Stage-1 Music Head checkpoint |
| `MERIC_STAGE2_CKPT` | Local Stage-2 flow decoder checkpoint |
| `MERIC_STABLE_AUDIO_DIR` | Local Stable Audio Open snapshot |
| `MERIC_MUQ_DIR` | Local MuQ-MuLan snapshot |
| `MERIC_STABLE_AUDIO_REPO` | Optional Stable Audio Open mirror |
| `MERIC_STABLE_AUDIO_REVISION` | Immutable revision for that mirror |
| `MERIC_MUQ_REPO` | Optional MuQ-MuLan mirror |
| `MERIC_MUQ_REVISION` | Immutable revision for that mirror |
Never commit a populated `.env` file or any provider credential.

## Model downloads

The first `MericPipeline.from_pretrained` call downloads the project weights and frozen upstream components.
All public defaults are pinned to immutable revisions.
The exact revisions, file sizes, and SHA-256 checksums are in [MODELS.md](MODELS.md).

For offline use, pre-populate the paths described in that document and export the matching environment variables.

## Verify the installation

Lightweight CPU checks do not download weights:

```bash
python -c "import meric; print(meric.__version__); print(meric.list_models())"
python -m pytest -q
ruff check meric scripts tests
ruff format --check meric scripts tests
```

Check the command surface:

```bash
meric models
meric generate --help
```

An end-to-end smoke test requires the Qwen3-VL environment, all weights, and a CUDA GPU:

```bash
meric generate \
    --image examples/images/sample_sunset.jpg \
    --seed 42 \
    --sample-steps 20 \
    --output-dir outputs/smoke
```

The reduced Stage-2 step count makes this a smoke test, not the paper configuration.
Paper generation uses 50 Stage-2 steps unless a table specifies otherwise.

## Common failures

### Qwen3-VL repository not found

Set `QWEN3VL_REPO` to the checkout created by `scripts/setup_qwen3vl.sh`.
The directory must contain `.venv/bin/python` and `models/Qwen3-VL-Embedding-2B` unless `QWEN3VL_MODEL` is set.

### CUDA out of memory

Generate one item at a time and close other GPU processes.
The full pipeline loads Qwen3-VL in a separate process and the audio models in the main process, so both environments need access to the intended GPU.

### Hub download failure

Confirm access to `https://huggingface.co/Audiofool/meric` and the upstream repositories listed in [MODELS.md](MODELS.md).
For an authenticated mirror, set `HF_TOKEN`, `MERIC_HF_REPO`, and `MERIC_HF_REVISION` as needed.

### Lightning or `pkg_resources` error

Confirm that `pytorch-lightning==1.9.3` and `setuptools==80.9.0` are installed.
Do not upgrade Lightning independently in the paper environment.
