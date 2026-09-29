# Usage

The supported inference surfaces are the `meric` command and `meric.MericPipeline`.
Both use the same two-stage implementation and pinned checkpoint resolver.

## Command line

List released Music Heads:

```bash
meric models
```

Generate from an image:

```bash
meric generate \
    --image examples/images/sample_sunset.jpg \
    --model meric-sft-v3 \
    --seed 42 \
    --output-dir outputs/image
```

Generate from text:

```bash
meric generate \
    --text "A calm piano melody with gentle rain" \
    --seed 42 \
    --output-dir outputs/text
```

Text follows the paper path:

```text
text -> Qwen3-VL-Embedding -> Music Head -> MuQ -> MericLDM -> audio
```

It does not bypass the Music Head or use MuQ-MuLan's text encoder as a shortcut.

Generate from video:

```bash
meric generate \
    --video path/to/clip.mp4 \
    --video-max-frames 8 \
    --seed 42 \
    --output-dir outputs/video
```

The default video path uniformly samples at most eight frames inside Qwen3-VL.
Increase `--video-max-frames` only when GPU memory permits and record the value with reported results.

Generate from a precomputed MuQ vector:

```bash
meric generate \
    --muq-npy path/to/embedding.npy \
    --seed 42 \
    --output-dir outputs/muq
```

The `.npy` file must have shape `[512]` or `[1, 512]` and contain finite floating-point values.
This mode skips Stage 1 but still uses the released Stage-2 decoder.

### Multiple variants

`-n N` uses consecutive seeds beginning at `--seed`.
The following command uses seeds 100, 101, 102, and 103 for both Music Head sampling and Stage-2 synthesis:

```bash
meric generate \
    --image photo.jpg \
    -n 4 \
    --seed 100 \
    --output-dir outputs/variants
```

### Generation controls

| Flag | Default | Meaning |
|---|---:|---|
| `--model` | `meric-sft-v3` | Released Music Head name |
| `--device` | `cuda:0` | Main Meric runtime device |
| `-n` | `1` | Number of variants |
| `--seed` | `42` | First deterministic generation seed |
| `--video-max-frames` | `8` | Maximum Qwen3-VL video frames |
| `--rdm-steps` | `20` | Music Head DDIM steps |
| `--rdm-guidance` | `2.0` | Music Head classifier-free guidance |
| `--guidance-scale` | `5.0` | Stage-2 classifier-free guidance |
| `--sample-steps` | `50` | Stage-2 ODE solver steps |
| `--sample-method` | `dopri5` | `euler`, `midpoint`, or `dopri5` |

The camera-ready generation configuration uses 20 Music Head steps, Music Head guidance 2.0, 50 Stage-2 steps, Stage-2 guidance 5.0, and `dopri5` unless a table states otherwise.

## Python API

Load a model once and reuse it:

```python
from meric import MericPipeline

pipe = MericPipeline.from_pretrained(
    "meric-sft-v3",
    device="cuda:0",
    rdm_steps=20,
    rdm_guidance=2.0,
    guidance_scale=5.0,
    sample_steps=50,
    sample_method="dopri5",
)

paths = pipe.generate(
    image="photo.jpg",
    seeds=[7, 11, 23],
    output_dir="outputs/python",
)
```

Exactly one of `image`, `video`, `text`, or `muq` must be provided.
Explicit `seeds` override `seed` and `n`.
Every call returns a list of `pathlib.Path` objects.

The one-shot helper is convenient for a single request but reloads all models on every call:

```python
import meric

paths = meric.generate(
    text="gentle strings under a starry sky",
    seed=42,
    output_dir="outputs/one-shot",
)
```

## Run manifests

Each high-level generation call writes `manifest.json` into the output directory.
The manifest records:

- the Meric model name, configured Hub revision, and expected checkpoint digests;
- configured Stable Audio Open and MuQ-MuLan revisions, plus the tested Qwen3-VL revisions;
- the input type, path or text, and input checksum where applicable;
- all generation settings and seeds;
- the Meric, Python, NumPy, PyTorch, CUDA, and device environment;
- output paths and SHA-256 checksums.

Seeds make runs reproducible within the same software, model, hardware, and numerical environment.
Bit-for-bit identity is not guaranteed across different GPU architectures or library builds.

## Low-level engine

The high-level API is recommended for normal use.
The lower-level engine remains available for pre-extracted conditions, bring-your-own CLIP Music Heads, RDM ensembles, and explicit seed lists:

Install OpenCLIP only when using that optional path:

```bash
pip install -e ".[clip]"
```

```bash
python -m meric.inference.run_pipeline --help
```

Example with a pre-extracted Qwen3-VL condition:

```bash
python -m meric.inference.run_pipeline \
    --vision-npy path/to/qwen_condition.npy \
    --backbone qwen3vl \
    --seeds 7,11,23 \
    --output-dir outputs/low-level
```

The low-level interface preserves underscore-style aliases for older commands.
New applications should use the high-level API.

## Retrieval experiments

Meric retrieval compares the Music Head's predicted MuQ vectors with ground-truth MuQ audio vectors using cosine similarity.
The camera-ready protocol uses Music Head guidance 10.0 on MelBench and MusicCaps retrieval, and guidance 2.0 on ARIA retrieval.

The public high-level facade focuses on generation.
The public artifact boundary, gallery protocol, and expected camera-ready numbers are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [RESULTS.md](RESULTS.md).

## Stage-1 training

Pretrain a Qwen3-VL Music Head from cached conditions and MuQ targets:

```bash
python scripts/train_pretrain.py \
    --manifest data/unified/meta/visual_train.jsonl \
    --val-manifest data/unified/meta/test_audioset_music.jsonl \
    --cond-key qwen3vl \
    --target-key muq \
    --model-channels 1536 \
    --num-res-blocks 12 \
    --beta-schedule squaredcos_cap_v2 \
    --prediction-type v_prediction \
    --ema-decay 0.9999 \
    --snr-gamma 5.0 \
    --seed 42 \
    --output-dir outputs/checkpoints/rdm_pretrain
```

Fine-tune the camera-ready Music Head recipe on ARIA with 15 percent AudioSet-Music replay:

```bash
python scripts/train_sft.py \
    --manifest data/unified/meta/sft_train.jsonl \
    --val-manifest data/unified/meta/test_sft.jsonl \
    --replay-manifest data/unified/meta/visual_train.jsonl \
    --replay-filter audioset_music \
    --replay-ratio 0.15 \
    --pretrained /path/to/pretrained_music_head.pth \
    --beta-schedule auto \
    --prediction-type auto \
    --ema-decay 0.9999 \
    --snr-gamma 5.0 \
    --seed 42 \
    --output-dir outputs/checkpoints/rdm_sft_v3
```

Run each script with `--help` for its complete argument set.
Dataset manifests and cached embeddings are described in [DATA_INVENTORY.md](DATA_INVENTORY.md).

## Stage-2 training

The Stage-2 reference module is `meric.workers.stableaudio_muq_flow.MericLDM`.
It exposes the Oobleck VAE and DiT initialization, MuQ-conditioned optimization step, and flow sampler used by the submitted system.
A complete Stage-2 training entry point and the prepared audio manifests are not yet part of the public artifact set.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the current boundary and required release inputs.

## Troubleshooting

For installation and model download issues, see [SETUP.md](SETUP.md).
For checkpoint details and checksums, see [MODELS.md](MODELS.md).
For the architecture and tensor shapes, see [ARCHITECTURE.md](ARCHITECTURE.md).
