# Data inventory and manifest format

This document describes the camera-ready datasets and the prepared-file contract used by the supported trainers.
Large datasets, audio, and embeddings are not stored in Git.

## Camera-ready dataset summary

Training sets are deduplicated by YouTube ID.
MuImage training examples are subsumed by AudioSet-Music after deduplication.

| Dataset | Input | Train | Test | Paper use | Public source |
|---|---|---:|---:|---|---|
| AudioSet-Music | video | 359K | 4,660 | Music Head pretraining, generation data source | Derived from AudioSet; release manifest pending |
| MelBench | image | 5,068 | 5,000 | Pretraining, generation, retrieval | Follow the MelBench project terms |
| EmoMV | video | 4,501 | 1,161 | Pretraining, generation | Follow the EmoMV project terms |
| MuImage | image | subsumed | 2,536 | Generation | Follow the MuImage project terms |
| MusicCaps | text | not used | 2,745 | Generation, retrieval | Public metadata; source audio remains subject to YouTube availability |
| ARIA | image | 4,000 | 1,000 | Fine-tuning, generation, retrieval | Public release pending |

The Music Head pretraining pool contains 369K cross-modal pairs from AudioSet-Music, MelBench, and EmoMV after deduplication.
The Flow Decoder training pool contains 100K FMA songs, 130K MusicSet clips, and 400K AudioSet-Music clips.

## Public release boundary

The repository contains validated loaders, Music Head training code, inference-time encoders, and exact result tables.
It does not contain raw media or prepared embeddings.

Full training and table regeneration require a separate artifact bundle with:

- immutable train and test manifests;
- stable sample identifiers and split definitions;
- normalized Qwen3-VL and MuQ embeddings;
- source-audio acquisition instructions;
- dataset-specific licenses and provenance;
- per-file checksums and aggregate counts.

ARIA additionally requires a dataset card covering source-image terms, caption generation, music generation, redistribution approval, and the 4K/1K split.
Until that release exists, pretrained inference is reproducible, while exact public retraining is not yet claimed.

## Directory convention

The loaders resolve relative paths against `--data-root`.
A recommended artifact layout is:

```text
data/
├── unified/
│   └── meta/
│       ├── visual_train.jsonl
│       ├── sft_train.jsonl
│       ├── test_audioset_music.jsonl
│       └── test_sft.jsonl
└── embeddings/
    ├── qwen3vl/
    ├── qwen3vl_text/
    └── muq/
```

Paths in manifests should be relative to the artifact root.
Do not publish workstation-specific absolute paths.

## Pretraining manifest schema

`UnifiedEmbeddingDataset` reads one JSON object per line.
A minimal visual record is:

```json
{
  "id": "stable-sample-id",
  "dataset": "audioset_music",
  "qwen3vl": "embeddings/qwen3vl/stable-sample-id.npy",
  "muq": "embeddings/muq/stable-sample-id.npy"
}
```

Optional condition keys include `clip`, `clip_keyframes`, `qwen3vl_keyframes`, and `qwen3vl_text`.
A list-valued condition or target is expanded into condition-target pairs by the loader.

The expected arrays are:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `qwen3vl` | `[2048]` | float32 | Normalized image or pooled-video condition |
| `qwen3vl_text` | `[2048]` | float32 | Normalized text condition |
| `clip` | `[1024]` | float32 | Optional normalized OpenCLIP condition |
| `muq` | `[512]` | float32 | Normalized MuQ-MuLan audio target |

Every released artifact validator should reject non-finite arrays and unexpected dimensions.

## ARIA fine-tuning manifest schema

`SFTMixedDataset` expects one record per image with a visual condition, a list of caption conditions, and MuQ segments grouped by caption index.

```json
{
  "id": "aria-000001",
  "dataset": "aria",
  "qwen3vl": "embeddings/qwen3vl/aria-000001.npy",
  "qwen3vl_text": [
    "embeddings/qwen3vl_text/aria-000001_c0.npy",
    "embeddings/qwen3vl_text/aria-000001_c1.npy",
    "embeddings/qwen3vl_text/aria-000001_c2.npy"
  ],
  "muq_segments": [
    "embeddings/muq/aria-000001_c0_s00.npy",
    "embeddings/muq/aria-000001_c1_s00.npy",
    "embeddings/muq/aria-000001_c2_s00.npy"
  ]
}
```

Caption and segment indices in file names are part of the loader contract.
Each epoch samples one MuQ segment for the image condition and one segment for each available caption condition.

## Embedding instructions

The released inference path uses the following exact Qwen3-VL instructions:

Visual input:

```text
Represent the emotional atmosphere and mood of this visual content for music generation.
```

Text input:

```text
Represent the emotional atmosphere and mood of this description for music generation.
```

Prepared training embeddings must use the same model revision, normalization, and instructions if they are intended to match released inference behavior.

## Split integrity requirements

A public artifact bundle should satisfy all of these checks:

1. No stable sample identifier occurs in both train and test splits.
2. YouTube-backed datasets are deduplicated by normalized YouTube ID before splitting.
3. Related image and video views of the same source stay in the same split.
4. Every manifest path exists and its digest matches the artifact index.
5. Every condition and target array has the documented dimension and contains only finite values.
6. Evaluation manifests preserve a deterministic row order and explicit denominator.
7. Missing or unavailable source media is reported, never silently replaced.

The release verifier in `scripts/release/verify_release.py` checks the source repository and model files.
A dataset-specific verifier must accompany the future data artifact because Git does not contain the files needed to perform these checks here.

## Licenses and provenance

Dataset terms are independent of the repository's Apache-2.0 source license.
Do not redistribute raw media, generated ARIA audio, or cached embeddings until the corresponding dataset card and upstream terms permit it.
See [NOTICE](../NOTICE) for the repository-level attribution summary and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the remaining publication records.
