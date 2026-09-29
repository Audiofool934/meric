import json

import numpy as np
import pytest

from meric.data.unified_dataset import SFTMixedDataset, UnifiedEmbeddingDataset


def save_embedding(path, dimension, *, value=1.0, dtype=np.float32):
    np.save(path, np.full(dimension, value, dtype=dtype))
    return path.name


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_unified_dataset_handles_mixed_scalar_and_list_paths(tmp_path):
    condition_a = save_embedding(tmp_path / "condition_a.npy", 2048, value=1)
    condition_b = save_embedding(tmp_path / "condition_b.npy", 2048, value=2)
    condition_c = save_embedding(tmp_path / "condition_c.npy", 2048, value=3)
    target_a = save_embedding(tmp_path / "target_a.npy", 512, value=4)
    target_b = save_embedding(tmp_path / "target_b.npy", 512, value=5)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {"id": "a", "dataset": "first", "qwen3vl": condition_a, "muq": target_a},
            {"id": "b", "dataset": "second", "qwen3vl": [condition_b, condition_c], "muq": target_b},
        ],
    )

    dataset = UnifiedEmbeddingDataset(manifest, preload=False, data_root=tmp_path)

    assert len(dataset) == 3
    assert dataset.dataset_counts == {"first": 1, "second": 2}
    condition, target = dataset[2]
    assert condition.shape == (2048,)
    assert target.shape == (512,)
    assert condition[0].item() == 3


@pytest.mark.parametrize(
    "array, message",
    [
        (np.ones(2047, dtype=np.float32), "must have shape"),
        (np.ones(2048, dtype=np.float64), "must use float32"),
        (np.full(2048, np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_unified_dataset_rejects_invalid_embeddings(tmp_path, array, message):
    np.save(tmp_path / "condition.npy", array)
    target = save_embedding(tmp_path / "target.npy", 512)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [{"id": "sample", "dataset": "test", "qwen3vl": "condition.npy", "muq": target}],
    )

    with pytest.raises(ValueError, match=message):
        UnifiedEmbeddingDataset(manifest, data_root=tmp_path)


def test_sft_dataset_validates_caption_groups_and_returns_typed_pairs(tmp_path):
    image = save_embedding(tmp_path / "item.npy", 2048)
    text_0 = save_embedding(tmp_path / "item_c0.npy", 2048, value=2)
    text_1 = save_embedding(tmp_path / "item_c1.npy", 2048, value=3)
    segment_0 = save_embedding(tmp_path / "item_c0_s00.npy", 512, value=4)
    segment_1 = save_embedding(tmp_path / "item_c1_s00.npy", 512, value=5)
    manifest = tmp_path / "sft.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "item",
                "dataset": "aria",
                "qwen3vl": image,
                "qwen3vl_text": [text_0, text_1],
                "muq_segments": [segment_0, segment_1],
            }
        ],
    )

    np.random.seed(42)
    dataset = SFTMixedDataset(manifest, data_root=tmp_path)

    assert len(dataset) == 3
    pair_types = sorted(dataset[index][2].item() for index in range(len(dataset)))
    assert pair_types == [0, 1, 1]
    for index in range(len(dataset)):
        condition, target, _ = dataset[index]
        assert condition.shape == (2048,)
        assert target.shape == (512,)


def test_sft_dataset_rejects_unpaired_caption_group(tmp_path):
    image = save_embedding(tmp_path / "item.npy", 2048)
    text = save_embedding(tmp_path / "item_c0.npy", 2048)
    segment_0 = save_embedding(tmp_path / "item_c0_s00.npy", 512)
    segment_1 = save_embedding(tmp_path / "item_c1_s00.npy", 512)
    manifest = tmp_path / "sft.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "item",
                "dataset": "aria",
                "qwen3vl": image,
                "qwen3vl_text": [text],
                "muq_segments": [segment_0, segment_1],
            }
        ],
    )

    with pytest.raises(ValueError, match="does not pair every caption group"):
        SFTMixedDataset(manifest, preload=False, data_root=tmp_path)
