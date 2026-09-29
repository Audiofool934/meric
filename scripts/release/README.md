# Release helpers

These maintainer tools validate the public release surface and update the existing Hugging Face model repository.
They do not create repositories or change repository visibility.
Use [docs/RELEASE_CHECKLIST.md](../../docs/RELEASE_CHECKLIST.md) as the launch gate for the exact release commit.
Public follow-up artifacts are tracked separately in [docs/ROADMAP.md](../../docs/ROADMAP.md).

Run the repository-level release checks:

```bash
python scripts/release/verify_release.py
```

Validate local checkpoint byte sizes and SHA-256 digests:

```bash
python scripts/release/verify_release.py --weights-dir /path/to/weights
```

Check Hugging Face authentication and the reviewed model card without writing anything:

```bash
python scripts/release/upload_to_hf.py --dry-run
```

Upload only the reviewed model card to the existing `Audiofool/meric` repository:

```bash
python scripts/release/upload_to_hf.py
```

Re-upload all three checkpoints only when necessary:

```bash
python scripts/release/upload_to_hf.py \
  --weights-dir /path/to/weights \
  --upload-weights \
  --dry-run

python scripts/release/upload_to_hf.py \
  --weights-dir /path/to/weights \
  --upload-weights
```

Checkpoint uploads refuse files whose size or digest differs from the release manifest in `meric/hub.py`.
The helper submits the reviewed model card and any selected checkpoints as one Hub commit.
Before redistributing the current Stage-2 checkpoint, maintainers must review both the Stability AI Community License and the CC BY-NC 4.0 terms recorded in `NOTICE` because the artifact serializes frozen MuQ-MuLan parameters.
After a write, validate the returned immutable Hub revision before changing the default pin in code.
