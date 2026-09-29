import json
import subprocess
from pathlib import Path

import numpy as np

from meric.inference import run_pipeline


def test_qwen_subprocess_uses_private_temp_request(monkeypatch, tmp_path):
    repo = tmp_path / "qwen"
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    observed = {}

    monkeypatch.setattr(run_pipeline, "_qwen3vl_runtime", lambda: (repo, python))

    def fake_run(command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        with request_path.open(encoding="utf-8") as handle:
            observed["request"] = json.load(handle)
        observed["request_path"] = request_path
        observed["output_path"] = output_path
        observed["kwargs"] = kwargs
        np.save(output_path, np.ones((1, 2048), dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    result = run_pipeline._extract_qwen3vl_inputs_via_subprocess(
        [{"text": "calm piano", "instruction": run_pipeline.TEXT_INSTRUCTION}],
        tmp_path,
        "cpu",
    )

    assert result.shape == (1, 2048)
    assert observed["request"]["inputs"][0]["text"] == "calm piano"
    assert observed["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not observed["request_path"].exists()
    assert not observed["output_path"].exists()


def test_qwen_subprocess_maps_logical_cuda_device(monkeypatch, tmp_path):
    repo = tmp_path / "qwen"
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    observed = {}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(run_pipeline, "_qwen3vl_runtime", lambda: (repo, python))

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        observed["visible_devices"] = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        np.save(output_path, np.ones((1, 2048), dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    run_pipeline._extract_qwen3vl_inputs_via_subprocess(
        [{"text": "calm piano", "instruction": run_pipeline.TEXT_INSTRUCTION}],
        tmp_path,
        "cuda:1",
    )

    assert observed["visible_devices"] == "5"
