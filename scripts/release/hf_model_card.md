---
license: other
license_name: multiple-upstream-model-licenses
license_link: https://huggingface.co/Audiofool/meric#licenses
base_model:
  - stabilityai/stable-audio-open-1.0
  - OpenMuQ/MuQ-MuLan-large
pipeline_tag: text-to-audio
tags:
  - music-generation
  - image-to-music
  - video-to-music
  - text-to-music
  - multimodal
  - cross-modal-retrieval
  - flow-matching
  - diffusion
library_name: meric
---

# Meric

Meric is the model release for **Meric: A Unified Framework for Multimodal Music Generation and Retrieval via Representation Space Anchoring**, accepted to ECCV 2026.

Meric generates 44.1 kHz music from images, videos, and text.
It first maps Qwen3-VL-Embedding features through a stochastic Music Head into a 512-dimensional MuQ music semantic anchor, then renders audio with a MuQ-conditioned flow decoder.
The same anchor supports cross-modal retrieval.

- [Code](https://github.com/Audiofool934/meric)
- [Paper](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/2169.pdf)
- [Project page](https://audiofool.blog/meric/)
- [Setup and reproducibility documentation](https://github.com/Audiofool934/meric/tree/main/docs)

The arXiv URL, proceedings pages, and DOI will be added when available.

The code repository is currently private while release preparation is completed.
The model weights are available now; the installation commands below require repository access until the code release.

## Files

| File | Registered name | Bytes | SHA-256 | Purpose |
|---|---|---:|---|---|
| `rdm_sft_v3.pth` | `meric-sft-v3` | 2,025,788,845 | `1f53e8d7bdae7309f48a4cecea2f0a3b92166ba8f09de37ad70a57ccbe595320` | Camera-ready Music Head and default model |
| `rdm_sft_instrumental.pth` | `meric-instrumental` | 2,025,788,845 | `09b290a6f31dccc3d0b521eb61b372bb368288d22203b592f0d6c62ca2d3b2c3` | Instrumental-focused Music Head |
| `mericldm.ckpt` | shared Stage 2 | 15,903,442,922 | `f43d8abc8d7a7aabe60ef68f29bce544641d236681f44f9ecadd33607fa4d47a` | MuQ-conditioned flow decoder |

These checkpoint contents were verified at revision `d96e2adcfa1daf9508beee26e9467e9b98e58681`.
The code repository records its current immutable Hub revision in `meric/hub.py` and `docs/MODELS.md`.

## Installation

Meric is tested on Linux with Python 3.10, PyTorch 2.5.1, and CUDA 12.1.
Qwen3-VL-Embedding runs in a separate Python 3.11 environment prepared by the repository helper.
All default project and upstream revisions are immutable.

```bash
git clone https://github.com/Audiofool934/meric.git
cd meric

python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .

python -m pip install uv==0.9.26
bash scripts/setup_qwen3vl.sh
export QWEN3VL_REPO="$PWD/.external/Qwen3-VL-Embedding"
```

## Usage

```bash
meric generate \
  --image examples/images/sample_sunset.jpg \
  --seed 42 \
  --output-dir outputs/sunset

meric generate \
  --text "A calm piano melody with gentle rain" \
  --seed 42 \
  --output-dir outputs/text

meric generate \
  --video path/to/clip.mp4 \
  --video-max-frames 8 \
  --seed 42 \
  --output-dir outputs/video
```

```python
from meric import MericPipeline

pipe = MericPipeline.from_pretrained("meric-sft-v3", device="cuda:0")
wavs = pipe.generate(image="photo.jpg", seeds=[7, 11, 23], output_dir="outputs/image")
```

Each run writes a `manifest.json` containing generation settings, exact seeds, model revisions, and input and output checksums.

## Training data

The Music Head is pretrained on 369K cross-modal pairs from AudioSet-Music, MelBench, and EmoMV, then fine-tuned on the 4K-image ARIA training split with 15 percent AudioSet replay.
The flow decoder is trained with MuQ conditions from 100K FMA songs, 130K MusicSet clips, and 400K AudioSet-Music clips.
See the paper and repository data documentation for the current artifact boundary, split definitions, and provenance requirements.

ARIA is not bundled with this model repository.
Its public release requires a separate dataset card and confirmation of redistribution terms.

## Camera-ready results

Lower FAD is better and higher Sem is better.

| Task | Dataset | FAD-MERT | FAD-CLAP | Sem |
|---|---|---:|---:|---:|
| Vision to music | MelBench | 1.55 | 0.127 | 74.7 |
| Vision to music | MuImage | 1.79 | 0.169 | 87.1 |
| Vision to music | EmoMV | 3.06 | 0.253 | 81.0 |
| Vision to music | ARIA | 3.30 | 0.247 | 79.6 |
| Text to music | MusicCaps | 1.87 | 0.110 | 74.2 |
| Text to music | ARIA | 2.14 | 0.155 | 74.5 |

Complete generation, retrieval, and ablation tables are transcribed in the code repository.

## Intended use

This release is intended for academic research, reproducibility studies, and creative prototyping of multimodal music generation and retrieval.
Generated audio should be reviewed before publication or downstream use.
Users are responsible for rights and consent associated with their inputs and outputs.

## Limitations

- Generation is stochastic and can vary substantially by seed.
- Semantic fit, musical quality, and cultural coverage vary by prompt and domain.
- The training corpora and frozen upstream encoders may introduce biases.
- The model is not designed for realistic singing voice generation.
- Evaluation relies partly on learned metrics and an LLM-based semantic judge, each with known limitations.
- End-to-end inference requires substantial GPU memory and roughly 30 GB of downloaded model artifacts.

## Licenses

The Meric source repository is Apache-2.0.
These model files are not Apache-2.0 licensed.
Stage 2 derives from Stable Audio Open 1.0 and is governed by the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md).
The current `mericldm.ckpt` also serializes frozen MuQ-MuLan parameters under the historical `clap.*` namespace; those parameters retain the upstream [CC BY-NC 4.0](https://github.com/tencent-ailab/MuQ/blob/main/LICENSE_weights) terms.
Qwen3-VL-Embedding and other dependencies retain their own terms.
Review the repository `NOTICE` before redistribution, deployment, or commercial use.

## Citation

```bibtex
@inproceedings{wang2026meric,
  title     = {Meric: A Unified Framework for Multimodal Music Generation and Retrieval via Representation Space Anchoring},
  author    = {Wang, Xihua and Wang, Yinbo and Zhang, Jingchao and Song, Ruihua},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

Xihua Wang, Yinbo Wang, and Jingchao Zhang contributed equally.
Ruihua Song is the corresponding author.
