# Architecture

Meric separates cross-modal understanding from acoustic synthesis through a shared music representation.
The frozen 512-dimensional MuQ space is the anchor between the two stages.

## End-to-end system

```text
                    Stage 1                                  Stage 2

 image ---------+
 video ---------+-> Qwen3-VL [2048] -> Music Head -> MuQ [512] -> condition projection
 text ----------+                        RDM / DDIM                 |
                                                                     v
                                                            Stable Audio DiT
                                                                     |
                                                            flow-matching ODE
                                                                     |
                                                            latent [64, 215]
                                                                     |
                                                             Oobleck decoder
                                                                     |
                                                            44.1 kHz stereo
```

Stage 1 learns cross-modal correspondence from paired multimodal and music data.
Stage 2 learns audio generation from unpaired music by conditioning on MuQ audio embeddings.
This separation lets one acoustic decoder serve all input modalities and lets the predicted anchor support both generation and retrieval.

## Public execution path

The high-level entry point is [`MericPipeline`](../meric/pipeline.py).

```python
from meric import MericPipeline

pipe = MericPipeline.from_pretrained("meric-sft-v3", device="cuda:0")
paths = pipe.generate(image="photo.jpg", n=3, seed=42, output_dir="outputs/architecture")
```

`from_pretrained` performs these operations:

1. Resolve the selected Music Head and shared Stage-2 checkpoint through [`meric.hub`](../meric/hub.py).
2. Resolve pinned Stable Audio Open and MuQ-MuLan snapshots.
3. Load the Stage-1 RDM checkpoint once with `weights_only=True` and restore its DDIM metadata.
4. Construct the Stage-2 VAE, DiT, MuQ condition encoder, and released state dict.
5. Defer Qwen3-VL loading until an image, video, or text request arrives.

The `meric` command calls the same class and does not maintain a separate inference implementation.

## Qwen3-VL isolation

Qwen3-VL-Embedding and the audio runtime require incompatible Python and PyTorch versions.
Meric isolates the backbone in a separate environment under `QWEN3VL_REPO`.

The main process writes a temporary JSON request containing native Qwen3-VL inputs.
It invokes [`_extract_qwen3vl.py`](../meric/inference/_extract_qwen3vl.py) with the external environment's Python, validates a finite `[N, 2048]` result, and removes both temporary request and result files.
No prompt is passed through a shell command.

The camera-ready preprocessing instructions are preserved exactly:

```text
Visual: Represent the emotional atmosphere and mood of this visual content for music generation.
Text:   Represent the emotional atmosphere and mood of this description for music generation.
```

Images are processed as Qwen3-VL image inputs.
Videos are processed as native video inputs with a configurable frame cap, defaulting to eight frames.
Text is processed by the same Qwen3-VL embedding model and then passed through the same Music Head as visual conditions.

## Stage 1: Music Head

The released Music Head is [`SimpleMLP`](../meric/models/rdm/latentmlp.py), a residual MLP conditioned on a 2048-dimensional Qwen3-VL vector and a diffusion timestep.

```text
condition:  [B, 2048]
noise:      [B, 512]
output:     [B, 512]
```

The camera-ready model uses:

- 1536 model channels;
- 12 residual blocks;
- squared-cosine diffusion schedule;
- v-prediction;
- exponential moving average;
- min-SNR loss weighting;
- classifier-free conditioning dropout;
- 20-step DDIM inference with guidance 2.0 for generation.

[`rdm_generate_muq`](../meric/inference/run_pipeline.py) runs the checkpoint's DDIM sampling configuration.
The released `meric-sft-v3` and `meric-instrumental` files both use this RDM path.

The Stage-1 seed initializes its 512-dimensional noise vector.
Different seeds therefore produce different plausible points in the music anchor space for the same condition.

### Bring-your-own CLIP path

The low-level engine retains the original CLIP ViT-H-14 condition path with dimension 1024.
No CLIP Music Head is distributed in the public model registry.
Users of that path must provide a compatible checkpoint explicitly.
Install its optional dependency with `pip install -e ".[clip]"`.

## Stage 2: MericLDM

[`MericLDM`](../meric/workers/stableaudio_muq_flow.py) maps a MuQ vector to audio.
It contains:

- the Oobleck VAE from Stable Audio Open;
- a Stable Audio DiT initialized from Stable Audio Open;
- a two-layer condition projection from MuQ dimension 512 to the DiT cross-attention dimension;
- a conditional optimal-transport flow path;
- an ODE solver with classifier-free guidance.

```text
MuQ condition:      [B, 512]
projected context:  [B, 1, cross_attention_dim]
initial latent:     [B, 64, 215]
decoded waveform:   [B, channels, samples]
sample rate:        44,100 Hz
```

The default inference path uses 50 solver steps, guidance 5.0, and `dopri5`.
The Stage-2 seed initializes the audio latent.
The high-level pipeline passes the same requested seed into Stage 1 and Stage 2 and records it in the run manifest.

The checkpoint loader transparently maps the historical `clap.*` key prefix to `muq_encoder.*`.
The underlying frozen encoder is MuQ-MuLan, and CLAP is not loaded by the inference path.

## Retrieval

Generation and retrieval share the Stage-1 prediction.
For retrieval, Meric does not run Stage 2.

```text
query modality -> Qwen3-VL -> Music Head -> predicted MuQ
candidate audio -> frozen MuQ encoder -> ground-truth MuQ
score -> cosine similarity
rank -> descending score
```

The camera-ready retrieval protocol reports both query-to-music and music-to-query Recall@K.
MelBench is evaluated as a one-to-one task.
ARIA is evaluated as a one-to-many task with approximately nine audio segments per visual query.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for gallery construction and [RESULTS.md](RESULTS.md) for expected values.

## Checkpoint and dependency resolution

The registry in [`meric/hub.py`](../meric/hub.py) pins every default remote dependency.
Project checkpoints resolve in this order:

1. Explicit environment override.
2. Local file under `$MERIC_HOME`.
3. Pinned Hugging Face Hub download.

Upstream snapshots have independent overrides:

- `MERIC_STABLE_AUDIO_DIR`
- `MERIC_MUQ_DIR`
- `QWEN3VL_REPO`
- `QWEN3VL_MODEL`

Exact revisions and SHA-256 checksums are listed in [MODELS.md](MODELS.md).

## Package layout

```text
meric/
├── __init__.py                         Lazy public exports
├── cli.py                              Public command line
├── pipeline.py                         High-level inference facade and manifests
├── hub.py                              Pinned model registry and download resolution
├── inference/
│   ├── run_pipeline.py                 Shared Stage-1 and Stage-2 engine
│   └── _extract_qwen3vl.py             Isolated multimodal embedding helper
├── models/
│   ├── rdm/latentmlp.py                Music Head residual MLP
│   └── stable_audio/                    DiT implementation
├── workers/
│   └── stableaudio_muq_flow.py          Stage-2 flow decoder
├── data/                                Cached-embedding training loader
└── utils/                               RDM sampling and training utilities
```

The supported camera-ready Stage-1 trainers are [`scripts/train_pretrain.py`](../scripts/train_pretrain.py) and [`scripts/train_sft.py`](../scripts/train_sft.py).
Internal experiment histories and incomplete post-submission branches are maintained outside the public release surface.

## Safety and reproducibility boundaries

Released PyTorch checkpoints are loaded with `weights_only=True`.
Input NumPy files are loaded with `allow_pickle=False`.
Remote model revisions are immutable by default.
Generation manifests record all requested seeds and relevant model revisions.

Numerical output can still vary across GPU architectures, CUDA versions, and solver implementations.
Reproducing paper tables also requires the exact dataset splits and external baseline environments described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
