from pathlib import Path

import huggingface_hub

from meric import hub


def test_release_dependencies_are_pinned():
    assert len(hub._DEFAULT_MERIC_REVISION) == 40
    assert len(hub.STABLE_AUDIO_REVISION) == 40
    assert len(hub.MUQ_MULAN_REVISION) == 40
    assert len(hub.QWEN3VL_CODE_REVISION) == 40
    assert len(hub.QWEN3VL_MODEL_REVISION) == 40
    assert all(spec["revision"] for spec in hub.MODELS.values())
    assert hub.STAGE2["revision"]
    for spec in [*hub.MODELS.values(), hub.STAGE2]:
        assert spec["bytes"] > 0
        assert len(spec["sha256"]) == 64


def test_mirror_does_not_inherit_default_revision(monkeypatch):
    monkeypatch.setenv("TEST_REPO", "mirror/example")
    monkeypatch.delenv("TEST_REVISION", raising=False)

    repo, revision = hub._repo_and_revision("TEST_REPO", "TEST_REVISION", "upstream/example", "abc123")

    assert repo == "mirror/example"
    assert revision is None


def test_resolve_forwards_revision(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.touch()
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("MERIC_HOME", str(tmp_path / "empty-cache"))

    resolved = hub._resolve("org/model", "weights.pth", revision="abc123")

    assert resolved == Path(tmp_path / "weights.pth")
    assert calls == [{"repo_id": "org/model", "filename": "weights.pth", "revision": "abc123"}]


def test_explicit_override_wins_without_downloading(monkeypatch, tmp_path):
    checkpoint = tmp_path / "local.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("TEST_MERIC_CHECKPOINT", str(checkpoint))

    def fail_download(**_kwargs):
        raise AssertionError("Hub download must not run when an override is present")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)

    assert (
        hub._resolve(
            "org/model",
            "remote.ckpt",
            env_override="TEST_MERIC_CHECKPOINT",
            revision="abc123",
        )
        == checkpoint
    )
