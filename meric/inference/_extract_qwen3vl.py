#!/usr/bin/env python3
"""Extract normalized Qwen3-VL embeddings for Meric.

This helper runs inside the external Qwen3-VL-Embedding environment. The
parent Meric process writes a JSON request containing native Qwen3-VL inputs,
then reads the resulting ``[N, 2048]`` NumPy array.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_request(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("The request must contain a non-empty 'inputs' list")
    if not all(isinstance(item, dict) for item in inputs):
        raise TypeError("Every Qwen3-VL input must be a JSON object")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON request path")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path")
    args = parser.parse_args()

    repo = Path(os.environ.get("QWEN3VL_REPO", Path.cwd())).expanduser().resolve()
    sys.path.insert(0, str(repo))

    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    model_path = Path(os.environ.get("QWEN3VL_MODEL", repo / "models" / "Qwen3-VL-Embedding-2B")).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3-VL model not found at {model_path}. Set QWEN3VL_MODEL or follow docs/SETUP.md.")

    inputs = _load_request(args.request)
    model = Qwen3VLEmbedder(model_name_or_path=str(model_path), torch_dtype=torch.bfloat16)

    embeddings = []
    for item in inputs:
        embedding = model.process([item], normalize=True)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.float().cpu().numpy()
        embeddings.append(np.asarray(embedding, dtype=np.float32))

    result = np.concatenate(embeddings, axis=0)
    if result.shape != (len(inputs), 2048):
        raise ValueError(f"Expected Qwen3-VL output shape {(len(inputs), 2048)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Qwen3-VL produced non-finite embeddings")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {result.shape} to {args.output}")


if __name__ == "__main__":
    main()
