"""Validated cached-embedding datasets for Music Head training."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

EMBEDDING_DIMS = {
    "clip": 1024,
    "clip_keyframes": 1024,
    "qwen3vl": 2048,
    "qwen3vl_keyframes": 2048,
    "qwen3vl_text": 2048,
    "muq": 512,
}


def _load_embedding(path, expected_dim):
    path = Path(path)
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"Could not load embedding {path}: {error}") from error
    if array.dtype != np.float32:
        raise ValueError(f"Embedding {path} must use float32, got {array.dtype}")
    if array.shape != (expected_dim,):
        raise ValueError(f"Embedding {path} must have shape ({expected_dim},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Embedding {path} contains non-finite values")
    return array


def _parallel_load_embeddings(items, *, desc, num_workers=32):
    """Load ``(path, expected_dim)`` items in stable order using parallel I/O."""
    if not items:
        return []
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    def load(indexed_item):
        index, (path, expected_dim) = indexed_item
        return index, _load_embedding(path, expected_dim)

    results = [None] * len(items)
    worker_count = min(num_workers, len(items))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(load, enumerate(items))
        for index, data in tqdm(iterator, total=len(items), desc=desc):
            results[index] = data
    return results


def _read_jsonl(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = []
    seen_ids = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Manifest entry {manifest_path}:{line_number} must be a JSON object")
            for key in ("id", "dataset"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"Manifest entry {manifest_path}:{line_number} requires a non-empty {key!r}")
            identity = (record["dataset"], record["id"])
            if identity in seen_ids:
                raise ValueError(f"Duplicate manifest identity at {manifest_path}:{line_number}: {identity}")
            seen_ids.add(identity)
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return records


def _path_values(record, key, *, manifest_path):
    if key not in record:
        raise ValueError(f"Record {record['id']!r} in {manifest_path} is missing {key!r}")
    value = record[key]
    values = value if isinstance(value, list) else [value]
    if not values or not all(isinstance(path, str) and path for path in values):
        raise ValueError(f"Record {record['id']!r} has an invalid {key!r} path value")
    return values


class UnifiedEmbeddingDataset(torch.utils.data.Dataset):
    """Load validated condition-target pairs from one JSONL manifest."""

    def __init__(
        self,
        manifest_path,
        cond_key="qwen3vl",
        target_key="muq",
        preload=True,
        data_root=".",
        filter_datasets=None,
        max_samples=None,
        preload_workers=32,
    ):
        if cond_key not in EMBEDDING_DIMS:
            raise ValueError(f"Unknown condition key {cond_key!r}; expected one of {sorted(EMBEDDING_DIMS)}")
        if target_key not in EMBEDDING_DIMS:
            raise ValueError(f"Unknown target key {target_key!r}; expected one of {sorted(EMBEDDING_DIMS)}")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        if preload_workers < 1:
            raise ValueError("preload_workers must be at least 1")

        self.manifest_path = Path(manifest_path)
        self.cond_key = cond_key
        self.target_key = target_key
        self.data_root = Path(data_root)
        self.preload = preload
        self.preload_workers = preload_workers

        selected_datasets = set(filter_datasets) if filter_datasets else None
        self.records = [
            record
            for record in _read_jsonl(self.manifest_path)
            if selected_datasets is None or record["dataset"] in selected_datasets
        ]
        if not self.records:
            raise ValueError(f"No records in {self.manifest_path} match the requested dataset filter")

        self.pairs = []
        for record_index, record in enumerate(self.records):
            conditions = _path_values(record, cond_key, manifest_path=self.manifest_path)
            targets = _path_values(record, target_key, manifest_path=self.manifest_path)
            for condition in conditions:
                for target in targets:
                    self.pairs.append((condition, target, record_index))
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        if not self.pairs:
            raise ValueError(f"No condition-target pairs were built from {self.manifest_path}")

        if preload:
            self._preload()

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.data_root / path

    def _preload(self):
        condition_dim = EMBEDDING_DIMS[self.cond_key]
        target_dim = EMBEDDING_DIMS[self.target_key]
        requirements = []
        for condition, target, _ in self.pairs:
            requirements.append((self._resolve(condition), condition_dim))
            requirements.append((self._resolve(target), target_dim))

        paths = {}
        for path, expected_dim in requirements:
            previous_dim = paths.setdefault(path, expected_dim)
            if previous_dim != expected_dim:
                raise ValueError(f"Embedding path is used with conflicting dimensions: {path}")
        resolved_paths = list(paths)
        loaded = _parallel_load_embeddings(
            [(path, paths[path]) for path in resolved_paths],
            desc="Preloading embeddings",
            num_workers=self.preload_workers,
        )
        cache = dict(zip(resolved_paths, loaded, strict=True))
        self._cond_data = np.stack([cache[self._resolve(condition)] for condition, _, _ in self.pairs])
        self._target_data = np.stack([cache[self._resolve(target)] for _, target, _ in self.pairs])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if self.preload:
            condition = self._cond_data[index]
            target = self._target_data[index]
        else:
            condition_path, target_path, _ = self.pairs[index]
            condition = _load_embedding(self._resolve(condition_path), EMBEDDING_DIMS[self.cond_key])
            target = _load_embedding(self._resolve(target_path), EMBEDDING_DIMS[self.target_key])
        return torch.from_numpy(condition.copy()), torch.from_numpy(target.copy())

    def get_record(self, index):
        """Return the manifest record associated with one expanded pair."""
        return self.records[self.pairs[index][2]]

    @property
    def dataset_counts(self):
        """Return expanded-pair counts by source dataset."""
        counts = {}
        for _, _, record_index in self.pairs:
            dataset = self.records[record_index]["dataset"]
            counts[dataset] = counts.get(dataset, 0) + 1
        return counts


class SFTMixedDataset(torch.utils.data.Dataset):
    """Sample one visual pair and one pair per text caption for each ARIA item."""

    def __init__(self, manifest_path, preload=True, data_root=".", image_only=False, preload_workers=32):
        if preload_workers < 1:
            raise ValueError("preload_workers must be at least 1")
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.preload = preload
        self.image_only = image_only
        self.preload_workers = preload_workers
        self.image_ids = []
        self.image_emb_paths = {}
        self.image_to_caps = {}
        self.text_emb_paths = {}
        self._caps_with_text = {}
        seen_image_ids = set()

        for record in _read_jsonl(self.manifest_path):
            image_id = record["id"]
            if image_id in seen_image_ids:
                raise ValueError(f"Duplicate SFT image id across datasets: {image_id!r}")
            seen_image_ids.add(image_id)
            image_paths = _path_values(record, "qwen3vl", manifest_path=self.manifest_path)
            if len(image_paths) != 1:
                raise ValueError(f"SFT record {image_id!r} must contain exactly one qwen3vl path")
            segment_paths = _path_values(record, "muq_segments", manifest_path=self.manifest_path)

            caption_segments = {}
            for segment_path in segment_paths:
                match = re.search(r"_c(\d+)_s\d+\.npy$", segment_path)
                if not match:
                    raise ValueError(f"SFT segment path does not encode caption and segment indices: {segment_path}")
                caption_index = int(match.group(1))
                caption_segments.setdefault(caption_index, []).append(segment_path)

            captions_with_text = {}
            if not image_only:
                text_paths = _path_values(record, "qwen3vl_text", manifest_path=self.manifest_path)
                for text_path in text_paths:
                    match = re.search(r"_c(\d+)\.npy$", text_path)
                    if not match:
                        raise ValueError(f"SFT text path does not encode a caption index: {text_path}")
                    caption_index = int(match.group(1))
                    key = (image_id, caption_index)
                    if key in self.text_emb_paths:
                        raise ValueError(f"Duplicate SFT caption index {caption_index} for {image_id!r}")
                    if caption_index not in caption_segments:
                        raise ValueError(f"SFT caption {caption_index} for {image_id!r} has no MuQ segments")
                    self.text_emb_paths[key] = text_path
                    captions_with_text[caption_index] = caption_segments[caption_index]
                if set(captions_with_text) != set(caption_segments):
                    raise ValueError(f"SFT record {image_id!r} does not pair every caption group with a text embedding")

            self.image_ids.append(image_id)
            self.image_emb_paths[image_id] = image_paths[0]
            self.image_to_caps[image_id] = caption_segments
            self._caps_with_text[image_id] = captions_with_text

        if preload:
            self._preload()
        self._build_pairs()
        if not self.pairs:
            raise ValueError(f"No SFT pairs were built from {self.manifest_path}")

        print(
            f"  SFTMixedDataset: {len(self.image_ids)} images, "
            f"{len(self.pairs)} pairs/epoch "
            f"({self.n_image_pairs} image + {self.n_text_pairs} text)"
        )

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.data_root / path

    def _preload(self):
        requirements = []
        for image_id in self.image_ids:
            requirements.append(("image", image_id, self.image_emb_paths[image_id], 2048))
            for caption_index, segment_paths in self.image_to_caps[image_id].items():
                text_key = (image_id, caption_index)
                if text_key in self.text_emb_paths:
                    requirements.append(("text", text_key, self.text_emb_paths[text_key], 2048))
                requirements.extend(("muq", path, path, 512) for path in segment_paths)

        paths = {}
        for _, _, relative_path, expected_dim in requirements:
            resolved = self._resolve(relative_path)
            previous_dim = paths.setdefault(resolved, expected_dim)
            if previous_dim != expected_dim:
                raise ValueError(f"Embedding path is used with conflicting dimensions: {resolved}")
        resolved_paths = list(paths)
        loaded = _parallel_load_embeddings(
            [(path, paths[path]) for path in resolved_paths],
            desc="Loading SFT embeddings",
            num_workers=self.preload_workers,
        )
        cache = dict(zip(resolved_paths, loaded, strict=True))

        self._image_embs = {}
        self._text_embs = {}
        self._muq_embs = {}
        for kind, key, relative_path, _ in requirements:
            data = cache[self._resolve(relative_path)]
            if kind == "image":
                self._image_embs[key] = data
            elif kind == "text":
                self._text_embs[key] = data
            else:
                self._muq_embs[key] = data

    def _build_pairs(self):
        image_pairs = []
        text_pairs = []
        for image_id in self.image_ids:
            all_segments = [path for paths in self.image_to_caps[image_id].values() for path in paths]
            image_segment = all_segments[np.random.randint(len(all_segments))]
            if self.preload:
                image_pairs.append((self._image_embs[image_id], self._muq_embs[image_segment], 0))
            else:
                image_pairs.append((self.image_emb_paths[image_id], image_segment, 0))

            for caption_index, segment_paths in self._caps_with_text[image_id].items():
                text_key = (image_id, caption_index)
                text_segment = segment_paths[np.random.randint(len(segment_paths))]
                if self.preload:
                    text_pairs.append((self._text_embs[text_key], self._muq_embs[text_segment], 1))
                else:
                    text_pairs.append((self.text_emb_paths[text_key], text_segment, 1))

        self.n_image_pairs = len(image_pairs)
        self.n_text_pairs = len(text_pairs)
        self.pairs = image_pairs + text_pairs
        np.random.shuffle(self.pairs)

    def resample(self):
        """Sample a new target segment for every image and caption pair."""
        self._build_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        condition, target, pair_type = self.pairs[index]
        if not self.preload:
            condition_dim = 2048
            condition = _load_embedding(self._resolve(condition), condition_dim)
            target = _load_embedding(self._resolve(target), 512)
        return (
            torch.from_numpy(condition.copy()),
            torch.from_numpy(target.copy()),
            torch.tensor(pair_type, dtype=torch.long),
        )


class ReplayWrapper(torch.utils.data.Dataset):
    """Add the replay pair-type label expected by the SFT trainer."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        condition, target = self.dataset[index]
        return condition, target, torch.tensor(2, dtype=torch.long)
