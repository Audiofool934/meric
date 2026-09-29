"""Smoke tests: the package imports and exposes its public API without weights.

Heavier checks (loading a model, generating audio) require the published
checkpoints and a GPU, and live outside this import-only suite.
"""

import importlib


def test_package_imports():
    import meric

    assert isinstance(meric.__version__, str)


def test_public_api_present():
    import meric

    assert callable(meric.MericPipeline.from_pretrained)
    assert callable(meric.generate)
    models = meric.list_models()
    assert "meric-sft-v3" in models


def test_core_modules_importable():
    for m in [
        "meric.hub",
        "meric.pipeline",
        "meric.inference.run_pipeline",
        "meric.models.rdm.latentmlp",
        "meric.utils.rdm_utils",
    ]:
        importlib.import_module(m)
