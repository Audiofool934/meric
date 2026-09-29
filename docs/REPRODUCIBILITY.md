# Reproducibility

Meric separates reproducibility into three levels: deterministic inference, model training, and regeneration of every camera-ready table.
This distinction prevents a runnable demo from being presented as full paper reproducibility.

## Current artifact coverage

| Artifact | Public status | What is reproducible |
|---|---|---|
| Source and environment | Available | Package build, CPU tests, lint, and pinned dependency installation |
| Released model weights | Available | Image, video, text, and direct-MuQ inference |
| Upstream frozen models | Available and pinned | Exact Stable Audio Open, MuQ-MuLan, and Qwen3-VL revisions |
| Music Head training code | Available | Pretraining and ARIA fine-tuning from prepared JSONL manifests |
| Flow decoder module | Partial | Architecture, optimization step, and sampler are inspectable; a complete training entry point is not yet released |
| Public training manifests and embeddings | Not yet released | Required for retraining from the same samples |
| ARIA dataset | Not yet released | Required for camera-ready fine-tuning and ARIA evaluation |
| Full metric and baseline environments | Not yet consolidated | Required to regenerate every comparison table |
| Semantic-judge raw outputs and human annotations | Not yet released | Required to regenerate the semantic validation result |

The source, pretrained inference path, pinned upstreams, and Music Head trainers form the supported repository release.
The remaining rows describe partial artifacts or release inputs, not hidden runtime requirements for pretrained inference.

## Immutable model inputs

The default runtime pins every downloaded model artifact.

| Component | Revision |
|---|---|
| Meric weights | `0a199f0f6a0c7f64f397a2192882883bfb2c7736` |
| Stable Audio Open 1.0 | `f21265c1e2710b3bd2386596943f0007f55f802e` |
| MuQ-MuLan-large | `2e01c796b71dca71b45251384c04cd7b237c9020` |
| Qwen3-VL-Embedding code | `8ce3aab1fbcc7b7143b094f0b2005dd89e7246b9` |
| Qwen3-VL-Embedding-2B | `2a50926d213628c727f38025982a76f655673f54` |

Exact checkpoint sizes and SHA-256 digests are listed in [MODELS.md](MODELS.md).
The current Stage-2 artifact contains frozen MuQ-MuLan parameters under a historical checkpoint prefix; [NOTICE](../NOTICE) records the resulting license boundary.
Verify local copies with:

```bash
python scripts/release/verify_release.py --weights-dir /path/to/weights
```

## Deterministic inference recipe

Install the two environments exactly as described in [SETUP.md](SETUP.md), then record the source revision:

```bash
git rev-parse HEAD
python -c "import meric; print(meric.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Generate one camera-ready-configuration sample:

```bash
meric generate \
  --image examples/images/sample_sunset.jpg \
  --model meric-sft-v3 \
  --rdm-steps 20 \
  --rdm-guidance 2.0 \
  --sample-steps 50 \
  --guidance-scale 5.0 \
  --sample-method dopri5 \
  --seed 42 \
  --output-dir outputs/reproduction
```

The output directory contains audio files and `manifest.json`.
The manifest records:

- the input path and SHA-256 digest;
- the exact Stage 1 and Stage 2 seeds;
- all sampling settings;
- configured Meric and upstream model revisions, expected project-checkpoint digests, and tested Qwen3-VL revisions;
- Meric, Python, NumPy, PyTorch, CUDA, and device environment details;
- each output file name and SHA-256 digest.

Repeated runs on identical software, model files, hardware, and CUDA libraries use the same requested random seeds.
Floating-point kernels can still differ across GPU architectures or CUDA and cuDNN versions, so byte-identical audio is not promised across different systems.
Use the manifest plus audio-level or embedding-level comparisons when validating another machine.

For an exact set of variants in Python, provide every seed explicitly:

```python
from meric import MericPipeline

pipe = MericPipeline.from_pretrained("meric-sft-v3", device="cuda:0")
pipe.generate(
    image="examples/images/sample_sunset.jpg",
    seeds=[7, 11, 23],
    output_dir="outputs/exact-seeds",
)
```

## Stage 1 pretraining

Prepared training manifests contain paths to normalized Qwen3-VL conditions and MuQ targets.
The expected JSONL schema is documented in [DATA_INVENTORY.md](DATA_INVENTORY.md).

The camera-ready Music Head recipe uses a squared-cosine noise schedule, v-prediction, EMA, min-SNR weighting, and seed 42.

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_pretrain.py \
  --manifest data/unified/meta/visual_train.jsonl \
  --val-manifest data/unified/meta/test_audioset_music.jsonl \
  --cond-key qwen3vl \
  --target-key muq \
  --output-dir outputs/checkpoints/rdm_pretrain \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --devices 0,1 \
  --seed 42
```

The script saves model state, optimizer state, scheduler and EMA state, training arguments, manifest and source-checkpoint SHA-256 digests, RNG state, and validation history.

## ARIA fine-tuning

Fine-tuning mixes image and text conditions from ARIA with 15 percent AudioSet-Music replay.

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py \
  --manifest data/unified/meta/sft_train.jsonl \
  --val-manifest data/unified/meta/test_sft.jsonl \
  --pretrained outputs/checkpoints/rdm_pretrain/best_model.pth \
  --replay-manifest data/unified/meta/visual_train.jsonl \
  --replay-filter audioset_music \
  --replay-ratio 0.15 \
  --output-dir outputs/checkpoints/rdm_sft_v3 \
  --epochs 80 \
  --batch-size 256 \
  --grad-accum 4 \
  --lr 1e-5 \
  --devices 0,1 \
  --seed 42
```

The exact training samples cannot be reconstructed until the ARIA release and prepared manifest bundle are published.

## Flow decoder training

The camera-ready Flow Decoder is trained on MuQ-conditioned music from 100K FMA songs, 130K MusicSet clips, and 400K AudioSet-Music clips.
The released `MericLDM` module contains the Flow Decoder architecture, optimization step, and sampler.

This stage requires the prepared audio metadata, MuQ features, and the approved dataset access paths.
Those artifacts are not distributed in Git.
The repository also does not yet expose a complete Stage-2 training entry point that assembles the dataset, trainer, logging, and checkpoint policy.
The released `mericldm.ckpt` is therefore the supported route for inference until the processed training manifest is published.

## Camera-ready evaluation protocol

The canonical numeric values are in [RESULTS.md](RESULTS.md).
The submitted setup uses:

- 20 DDIM Music Head steps;
- guidance 2.0 for generation and ARIA retrieval;
- guidance 10.0 for MelBench and MusicCaps retrieval;
- 50 Stage 2 flow-sampling steps for generation;
- generation metrics in MERT, CLAP, and VGGish spaces;
- retrieval by cosine similarity in the predicted and ground-truth MuQ anchor space;
- the paper's six-way semantic-judge protocol for Sem validation.

Exact table regeneration additionally needs test manifests with stable sample identifiers, ground-truth audio, baseline outputs, feature-extractor revisions, and the raw semantic-judge records.
These must be published together so that row ordering, sample exclusions, and denominators are auditable.

## Required release records

Before claiming full paper reproducibility, the public artifact set must add:

1. ARIA dataset URL, license, dataset card, and immutable release revision.
2. Prepared train and test JSONL manifests with sample counts and checksums.
3. Public preprocessing commands for every condition and target embedding.
4. Baseline environment locks and generated-output manifests.
5. Metric extractor revisions and per-sample evaluation manifests.
6. Semantic-judge prompts, raw machine outputs, anonymized human annotations, and analysis script.
7. Training hardware, wall-clock time, peak memory, and final checkpoint-selection rule.
8. An arXiv identifier and official proceedings metadata when assigned.

Until these records exist, the repository accurately supports pretrained inference, complete Music Head training-code inspection, and partial Flow Decoder training-code inspection.
It does not claim that every paper table can be regenerated using public assets alone.
