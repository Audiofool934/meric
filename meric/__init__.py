"""Meric - multimodal music generation and retrieval through a MuQ semantic anchor.

Public API (lazily imported so ``import meric`` stays light):

    from meric import MericPipeline
    pipe = MericPipeline.from_pretrained("meric-sft-v3", device="cuda:0")
    wavs = pipe.generate(image="photo.jpg", n=3)

    # or the one-shot helper
    import meric
    meric.generate(image="photo.jpg", output_dir="outputs/")

    # list available pretrained models
    meric.list_models()
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["MericPipeline", "generate", "list_models"]


def __getattr__(name):  # PEP 562 - defer heavy imports (torch/diffusers) until first use
    if name in ("MericPipeline", "generate"):
        return getattr(import_module("meric.pipeline"), name)
    if name == "list_models":
        return import_module("meric.hub").list_models
    raise AttributeError(f"module 'meric' has no attribute {name!r}")
