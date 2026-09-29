#!/usr/bin/env python3
"""Run offline checks against the public Meric release surface."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from meric import hub  # noqa: E402

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "docs/ARCHITECTURE.md",
    "docs/MODELS.md",
    "docs/ROADMAP.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS.md",
    "docs/SETUP.md",
    "docs/USAGE.md",
    "scripts/release/hf_model_card.md",
)

FORBIDDEN_TEXT = {
    "MERIC" + "-TODO": "placeholder model repository",
    "MERIC " + "authors": "placeholder author list",
    "github.com/" + "TODO": "placeholder GitHub URL",
    "public arXiv " + "release available": "unverified arXiv availability claim",
    "The public arXiv " + "release contains": "unverified arXiv availability claim",
    "." + "claude/worktrees": "private worktree path",
    "/data-01/" + "yinbo": "private filesystem path",
    "http://" + "mirrors.aliyun.com": "insecure package index",
    "scripts/" + "inference/run_pipeline.py": "removed inference path",
    "inputs/" + "images": "removed sample-input path",
}

BINARY_SUFFIXES = {".ckpt", ".mp3", ".npy", ".npz", ".pth", ".pt", ".wav", ".zip"}
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
QWEN_LOCK_SHA256 = "1f7b51793717d81984a081eafe097ef41f5a3f7912953a0ea8e158232af54609"
QWEN_UV_VERSION = "0.9.26"


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [path for item in result.stdout.decode().split("\0") if item and (path := ROOT / item).is_file()]


def _checkpoint_specs() -> dict[str, dict]:
    specs = {spec["filename"]: spec for spec in hub.MODELS.values()}
    specs[hub.STAGE2["filename"]] = hub.STAGE2
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_required(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"required file is missing: {relative}")


def _check_tracked_files(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"binary artifact is tracked in Git: {relative}")
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"tracked file exceeds 5 MiB: {relative}")


def _check_text(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() not in {"", ".cff", ".html", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token, description in FORBIDDEN_TEXT.items():
            if token in content:
                errors.append(f"{path.relative_to(ROOT)} contains {description}: {token}")


def _check_markdown_links(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        content = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(content):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"broken relative link in {path.relative_to(ROOT)}: {raw_target}")


def _check_metadata(errors: list[str]) -> None:
    expected_title = (
        "Meric: A Unified Framework for Multimodal Music Generation and Retrieval via Representation Space Anchoring"
    )
    expected_authors = ("Xihua Wang", "Yinbo Wang", "Jingchao Zhang", "Ruihua Song")
    surfaces = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "CITATION.cff": (ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        "model card": (ROOT / "scripts/release/hf_model_card.md").read_text(encoding="utf-8"),
    }
    for name, content in surfaces.items():
        if expected_title not in content:
            errors.append(f"{name} does not contain the canonical paper title")
        for author in expected_authors:
            if name == "CITATION.cff":
                given, family = author.split()
                if f'given-names: "{given}"' in content and f'family-names: "{family}"' in content:
                    continue
            elif author in content:
                continue
            errors.append(f"{name} does not contain author metadata for {author}")

    if '__version__ = "0.1.0"' not in (ROOT / "meric/__init__.py").read_text(encoding="utf-8"):
        errors.append("meric.__version__ is not 0.1.0")
    if 'version = "0.1.0"' not in (ROOT / "pyproject.toml").read_text(encoding="utf-8"):
        errors.append("pyproject.toml version is not 0.1.0")

    model_docs = {
        "docs/MODELS.md": (ROOT / "docs/MODELS.md").read_text(encoding="utf-8"),
        "model card": (ROOT / "scripts/release/hf_model_card.md").read_text(encoding="utf-8"),
    }
    for filename, spec in _checkpoint_specs().items():
        for name, content in model_docs.items():
            for value in (filename, f"{spec['bytes']:,}", spec["sha256"]):
                if value not in content:
                    errors.append(f"{name} is missing checkpoint metadata {value}")

    license_surfaces = {
        "NOTICE": (ROOT / "NOTICE").read_text(encoding="utf-8"),
        "docs/MODELS.md": model_docs["docs/MODELS.md"],
        "model card": model_docs["model card"],
    }
    for name, content in license_surfaces.items():
        for term in ("Stability AI Community License", "CC BY-NC 4.0", "MuQ-MuLan"):
            if term not in content:
                errors.append(f"{name} is missing model-license disclosure: {term}")

    notice = license_surfaces["NOTICE"]
    for term in ("improved-diffusion", "latent-diffusion", "Hugging Face Diffusers", "MIT License"):
        if term not in notice:
            errors.append(f"NOTICE is missing adapted-source attribution: {term}")

    qwen_setup = (ROOT / "scripts/setup_qwen3vl.sh").read_text(encoding="utf-8")
    for value in (
        hub.QWEN3VL_CODE_REVISION,
        hub.QWEN3VL_MODEL_REVISION,
        f'QWEN_LOCK_SHA256="{QWEN_LOCK_SHA256}"',
        f'QWEN_UV_VERSION="{QWEN_UV_VERSION}"',
    ):
        if value not in qwen_setup:
            errors.append(f"scripts/setup_qwen3vl.sh is missing pinned Qwen value: {value}")
    if "--require-hashes" not in qwen_setup or "https://pypi.org/simple" not in qwen_setup:
        errors.append("scripts/setup_qwen3vl.sh does not enforce hashed HTTPS dependency installation")


def _check_dependency_mirror(errors: list[str]) -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependency_block = pyproject.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    project_dependencies = set(re.findall(r'^\s*"([^"]+)",?$', dependency_block, flags=re.MULTILINE))
    requirement_dependencies = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if project_dependencies != requirement_dependencies:
        missing = sorted(project_dependencies - requirement_dependencies)
        extra = sorted(requirement_dependencies - project_dependencies)
        errors.append(f"requirements.txt differs from pyproject.toml: missing={missing}, extra={extra}")


def _check_workflow_pins(errors: list[str]) -> None:
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        for line_number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), start=1):
            match = re.search(r"\buses:\s*([^\s]+)", line)
            if not match or match.group(1).startswith("./"):
                continue
            reference = match.group(1).rsplit("@", 1)[-1]
            if not re.fullmatch(r"[0-9a-f]{40}", reference):
                errors.append(f"{workflow.relative_to(ROOT)}:{line_number} action is not pinned to a full commit SHA")


def _check_weights(directory: Path, errors: list[str]) -> None:
    for name, spec in _checkpoint_specs().items():
        path = directory / name
        if not path.is_file():
            errors.append(f"checkpoint is missing: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != spec["bytes"]:
            errors.append(f"{name} has {actual_size} bytes, expected {spec['bytes']}")
            continue
        actual_digest = _sha256(path)
        if actual_digest != spec["sha256"]:
            errors.append(f"{name} SHA-256 is {actual_digest}, expected {spec['sha256']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, help="Also verify all three local checkpoint files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    files = _tracked_files()

    _check_required(errors)
    _check_tracked_files(files, errors)
    _check_text(files, errors)
    _check_markdown_links(files, errors)
    _check_metadata(errors)
    _check_dependency_mirror(errors)
    _check_workflow_pins(errors)
    if args.weights_dir:
        _check_weights(args.weights_dir.expanduser().resolve(), errors)

    if errors:
        print(f"Release verification failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Release verification passed for {len(files)} tracked files.")
    if args.weights_dir:
        print(f"Verified checkpoint sizes and SHA-256 digests in {args.weights_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
