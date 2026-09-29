#!/usr/bin/env python3
"""Publish the reviewed Meric model card and optional verified checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meric import hub  # noqa: E402

DEFAULT_REPO = os.environ.get("MERIC_HF_REPO", "Audiofool/meric")
MODEL_CARD = Path(__file__).with_name("hf_model_card.md")


def _checkpoint_specs() -> dict[str, dict]:
    specs = {spec["filename"]: spec for spec in hub.MODELS.values()}
    specs[hub.STAGE2["filename"]] = hub.STAGE2
    return specs


def _source_paths(weights_dir: Path) -> dict[str, Path]:
    paths = {name: weights_dir / name for name in _checkpoint_specs()}
    if value := os.environ.get("MERIC_SFT_V3_CKPT"):
        paths["rdm_sft_v3.pth"] = Path(value)
    if value := os.environ.get("MERIC_INSTRUMENTAL_CKPT"):
        paths["rdm_sft_instrumental.pth"] = Path(value)
    if value := os.environ.get("MERIC_STAGE2_CKPT"):
        paths["mericldm.ckpt"] = Path(value)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint(name: str, path: Path, spec: dict) -> None:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing checkpoint for {name}: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec["bytes"]:
        raise SystemExit(f"ERROR: {name} has {actual_size} bytes, expected {spec['bytes']}")
    actual_digest = _sha256(path)
    if actual_digest != spec["sha256"]:
        raise SystemExit(f"ERROR: {name} SHA-256 is {actual_digest}, expected {spec['sha256']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="Existing Hugging Face model repository")
    parser.add_argument("--revision", default="main", help="Target branch or revision")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(os.environ.get("MERIC_HOME", Path.home() / ".cache" / "meric")),
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--upload-weights", action="store_true", help="Upload all three checkpoints after checksum verification"
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and authentication without uploading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not MODEL_CARD.is_file():
        raise SystemExit(f"ERROR: model card not found: {MODEL_CARD}")

    sources = _source_paths(args.weights_dir)
    if args.upload_weights:
        for name, spec in _checkpoint_specs().items():
            print(f"Verifying {name}: {sources[name]}")
            _verify_checkpoint(name, sources[name], spec)

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        identity = api.whoami().get("name", "unknown")
        before = api.repo_info(args.repo_id, repo_type="model", revision=args.revision)
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot access {args.repo_id}: {type(exc).__name__}: {exc}") from exc

    print(f"Authenticated as: {identity}")
    print(f"Target: https://huggingface.co/{args.repo_id}/tree/{args.revision}")
    print(f"Current revision: {before.sha}")
    print(f"Model card: {MODEL_CARD}")
    print(f"Checkpoint upload: {'enabled' if args.upload_weights else 'disabled'}")

    if args.dry_run:
        print("Dry run complete. Nothing was uploaded.")
        return 0

    from huggingface_hub import CommitOperationAdd

    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(MODEL_CARD))]
    if args.upload_weights:
        operations.extend(
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in sources.items()
        )

    print(f"Creating one reviewed commit with {len(operations)} file operation(s)")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        operations=operations,
        commit_message="Publish verified Meric release artifacts" if args.upload_weights else "Update Meric model card",
    )

    after = api.repo_info(args.repo_id, repo_type="model", revision=args.revision)
    print(f"Published revision: {after.sha}")
    print("Update meric/hub.py only after validating this immutable revision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
