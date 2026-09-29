import numpy as np
import pytest
import torch

from meric.models.rdm.latentmlp import SimpleMLP
from meric.utils.rdm_utils import (
    WarmupCosineScheduler,
    atomic_torch_save,
    capture_rng_state,
    normalize_gradients,
    restore_rng_state,
    sha256_file,
)


def test_music_head_preserves_batch_dimension_for_single_sample():
    model = SimpleMLP(
        in_channels=4,
        time_embed_dim=8,
        model_channels=8,
        bottleneck_channels=8,
        out_channels=4,
        num_res_blocks=2,
        use_context=True,
        context_channels=6,
    )

    output = model(
        x=torch.ones(1, 4, 1, 1),
        timesteps=torch.tensor([10]),
        context=torch.ones(1, 6),
    )

    assert output.shape == (1, 4)


def test_music_head_requires_context_when_configured():
    model = SimpleMLP(4, 8, 8, 8, 4, 1, use_context=True, context_channels=6)

    with pytest.raises(ValueError, match="context is required"):
        model(x=torch.ones(1, 4), timesteps=torch.tensor([0]))


def test_warmup_cosine_scheduler_sets_first_epoch_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=6,
        base_lr=1.0,
        min_lr=0.1,
    )

    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    assert observed[0] == pytest.approx(0.5)
    assert observed[1] == pytest.approx(1.0)
    assert observed[2] == pytest.approx(1.0)
    assert observed[-1] == pytest.approx(0.1)


def test_warmup_cosine_scheduler_restores_exact_state():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=6, base_lr=1.0, min_lr=0.1)
    scheduler.step()
    scheduler.step()

    resumed_parameter = torch.nn.Parameter(torch.tensor(0.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=1.0)
    resumed = WarmupCosineScheduler(
        resumed_optimizer,
        warmup_epochs=2,
        total_epochs=6,
        base_lr=1.0,
        min_lr=0.1,
    )
    resumed.load_state_dict(scheduler.state_dict())

    assert resumed.current_epoch == scheduler.current_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_normalize_gradients_uses_accumulated_sample_count():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(10.0)

    normalize_gradients([parameter], sample_count=4)

    assert parameter.grad.item() == pytest.approx(2.5)


def test_rng_state_round_trips_through_weights_only_checkpoint(tmp_path):
    original_state = capture_rng_state()
    try:
        torch.manual_seed(123)
        np.random.seed(123)
        state = capture_rng_state()
        expected_torch = torch.rand(4)
        expected_numpy = np.random.rand(4)

        path = tmp_path / "rng.pth"
        atomic_torch_save({"rng_state": state}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert restore_rng_state(loaded["rng_state"])

        assert torch.equal(torch.rand(4), expected_torch)
        assert np.array_equal(np.random.rand(4), expected_numpy)
    finally:
        restore_rng_state(original_state)


def test_sha256_file_uses_file_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"meric")

    assert sha256_file(path, chunk_size=2) == "a0d68072a5367fcd0be41642552582b07c335944dca4df5a69a976a9b4b37ddb"
