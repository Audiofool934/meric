import json
from pathlib import Path

import numpy as np
import pytest
import torch

from meric import pipeline as pipeline_module


def make_pipeline():
    pipe = pipeline_module.MericPipeline.__new__(pipeline_module.MericPipeline)
    pipe.rdm_model = object()
    pipe.rdm_sampler = object()
    pipe.meric_ldm = object()
    pipe.backbone = "qwen3vl"
    pipe.context_dim = 2048
    pipe.device = torch.device("cpu")
    pipe.rdm_steps = 20
    pipe.rdm_guidance = 2.0
    pipe.rdm_scale_factor = 10.0
    pipe._clip = None
    return pipe


def install_fake_generation(monkeypatch, tmp_path):
    stage1_seeds = []
    stage2_calls = []

    def fake_rdm(_model, _sampler, condition, _device, **kwargs):
        assert condition.shape == (1, 2048)
        stage1_seeds.append(kwargs["seed"])
        return torch.full((1, 512), float(kwargs["seed"]))

    def fake_audio(_model, embedding, output_dir, name, _device, seed):
        stage2_calls.append((name, seed, embedding.clone()))
        path = Path(output_dir) / f"{name}_00.wav"
        path.touch()
        return path

    monkeypatch.setattr(pipeline_module._engine, "rdm_generate_muq", fake_rdm)
    monkeypatch.setattr(pipeline_module._engine, "generate_audio", fake_audio)
    return stage1_seeds, stage2_calls


def test_text_uses_music_head_and_all_requested_seeds(monkeypatch, tmp_path):
    pipe = make_pipeline()
    monkeypatch.setattr(
        pipe,
        "_text_embedding",
        lambda text, _output_dir: np.ones((1, 2048), dtype=np.float32),
    )
    stage1_seeds, stage2_calls = install_fake_generation(monkeypatch, tmp_path)

    outputs = pipe.generate(text="calm piano / rain", seeds=[7, 11], output_dir=tmp_path)

    assert stage1_seeds == [7, 11]
    assert [call[1] for call in stage2_calls] == [7, 11]
    assert [path.name for path in outputs] == [
        "calm_piano_rain_seed7_00.wav",
        "calm_piano_rain_seed11_00.wav",
    ]


def test_video_uses_qwen_condition_and_frame_limit(monkeypatch, tmp_path):
    pipe = make_pipeline()
    video = tmp_path / "sample.mp4"
    video.touch()
    observed = {}

    def fake_video(path, _output_dir, max_frames):
        observed.update(path=path, max_frames=max_frames)
        return np.ones((1, 2048), dtype=np.float32)

    monkeypatch.setattr(pipe, "_video_embedding", fake_video)
    stage1_seeds, stage2_calls = install_fake_generation(monkeypatch, tmp_path)

    outputs = pipe.generate(video=video, seed=23, output_dir=tmp_path, video_max_frames=6)

    assert observed == {"path": video.resolve(), "max_frames": 6}
    assert stage1_seeds == [23]
    assert stage2_calls[0][1] == 23
    assert outputs[0].name == "sample_00.wav"


def test_muq_variants_are_stage2_deterministic(monkeypatch, tmp_path):
    pipe = make_pipeline()
    stage1_seeds, stage2_calls = install_fake_generation(monkeypatch, tmp_path)

    outputs = pipe.generate(muq=np.ones(512, dtype=np.float32), n=3, seed=40, output_dir=tmp_path)

    assert stage1_seeds == []
    assert [call[1] for call in stage2_calls] == [40, 41, 42]
    assert [path.name for path in outputs] == [
        "muq_seed40_00.wav",
        "muq_seed41_00.wav",
        "muq_seed42_00.wav",
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in manifest["outputs"]] == [40, 41, 42]
    assert manifest["input"]["shape"] == [1, 512]
    assert manifest["input"]["dtype"] == "float32"
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["environment"]["torch"] == torch.__version__
    assert all(len(item["sha256"]) == 64 for item in manifest["outputs"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"image": "a.jpg", "text": "prompt"}, "exactly one"),
        ({"text": ""}, "non-empty"),
        ({"muq": np.zeros((2, 512))}, "one MuQ embedding"),
        ({"muq": np.ones(512, dtype=np.complex64)}, "real numeric"),
        ({"text": "prompt", "n": 0}, "at least 1"),
        ({"text": "prompt", "n": -1, "seeds": [7]}, "at least 1"),
        ({"text": "prompt", "seeds": [-1]}, "between 0"),
        ({"text": "prompt", "seeds": [1.5]}, "must be an integer"),
        ({"text": "prompt", "seeds": [7, 7]}, "duplicates"),
    ],
)
def test_invalid_generation_requests_fail_early(tmp_path, kwargs, message):
    pipe = make_pipeline()
    with pytest.raises((TypeError, ValueError, FileNotFoundError), match=message):
        pipe.generate(output_dir=tmp_path, **kwargs)


def test_one_shot_rejects_bad_muq_file_before_model_loading(monkeypatch, tmp_path):
    embedding = tmp_path / "bad_muq.npy"
    np.save(embedding, np.ones(16, dtype=np.float32))

    def fail_loading(*_args, **_kwargs):
        raise AssertionError("model loading must not start for an invalid embedding")

    monkeypatch.setattr(pipeline_module.MericPipeline, "from_pretrained", fail_loading)

    with pytest.raises(ValueError, match="one MuQ embedding"):
        pipeline_module.generate(muq=embedding, output_dir=tmp_path)
