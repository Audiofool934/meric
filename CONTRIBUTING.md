# Contributing to Meric

Thank you for helping improve Meric.
The primary maintenance goal is preserving reproducibility of the published model while making the supported paths clearer and more reliable.

## Report an issue

Use [GitHub Issues](https://github.com/Audiofool934/meric/issues) for reproducible bugs, questions, and focused feature requests.
Include the source revision, operating system, Python and PyTorch versions, GPU and CUDA versions, command, complete traceback, and minimal input when possible.

Report security problems privately by following [SECURITY.md](SECURITY.md).

## Development setup

Meric targets Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

The Qwen3-VL environment is needed only for end-to-end image, video, or text inference.
Follow [docs/SETUP.md](docs/SETUP.md) when a change touches that boundary.

## Required checks

Run these commands from the repository root:

```bash
ruff check meric scripts tests
ruff format --check meric scripts tests
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/release/verify_release.py
```

CI enforces the same source, test, package, and release-surface checks.

When a change affects GPU inference, also run the closest end-to-end command from [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and attach its `manifest.json`.

## Pull requests

1. Branch from `main` and keep the change focused.
2. Explain the user-visible behavior and the reason for the change.
3. State which checks ran and which hardware-dependent checks could not run.
4. Call out any change to preprocessing, sampling defaults, tensor dimensions, model resolution, or reported metrics.
5. Update every affected documentation and metadata surface in the same pull request.

Behavior changes that would invalidate the camera-ready model should use an explicit opt-in path or a separately versioned model instead of silently replacing released defaults.

## Large artifacts and credentials

Do not commit checkpoints, datasets, generated audio, extracted embeddings, API keys, tokens, or populated environment files.
Host approved model artifacts on Hugging Face and record their immutable revision, byte size, SHA-256 digest, license, and provenance.

## License

Contributions to this repository are accepted under the [Apache License 2.0](LICENSE).
Model weights and third-party datasets retain their separate upstream terms.
