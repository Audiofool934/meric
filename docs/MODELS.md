# Models and checkpoints

Meric combines two project-trained Music Heads, one shared flow decoder, and three frozen upstream components.
All default downloads are pinned to immutable revisions in [`meric/hub.py`](../meric/hub.py).

## Released Meric weights

The model repository is [Audiofool/meric](https://huggingface.co/Audiofool/meric).
The tested Hub revision is `0a199f0f6a0c7f64f397a2192882883bfb2c7736`.

| File | Registered name | Bytes | SHA-256 |
|---|---|---:|---|
| `rdm_sft_v3.pth` | `meric-sft-v3` | 2,025,788,845 | `1f53e8d7bdae7309f48a4cecea2f0a3b92166ba8f09de37ad70a57ccbe595320` |
| `rdm_sft_instrumental.pth` | `meric-instrumental` | 2,025,788,845 | `09b290a6f31dccc3d0b521eb61b372bb368288d22203b592f0d6c62ca2d3b2c3` |
| `mericldm.ckpt` | shared Stage 2 | 15,903,442,922 | `f43d8abc8d7a7aabe60ef68f29bce544641d236681f44f9ecadd33607fa4d47a` |

`meric-sft-v3` is the camera-ready paper model and the default.
`meric-instrumental` uses the same architecture and decoder but was fine-tuned on vocal-filtered data for cleaner instrumental output.

Both Music Heads use Qwen3-VL-Embedding conditions with dimension 2048 and generate MuQ anchor vectors with dimension 512.
They share `mericldm.ckpt` for MuQ-to-audio generation.

The published `mericldm.ckpt` also serializes 805 frozen MuQ-MuLan state entries under the historical `clap.*` namespace.
Those entries retain the upstream CC BY-NC 4.0 terms even though the runtime also initializes MuQ-MuLan from its pinned upstream snapshot.
Removing the duplicated frozen state in a future checkpoint would require a new Hub revision, byte size, and SHA-256 digest.

List the registry from Python or the command line:

```python
import meric

print(meric.list_models())
```

```bash
meric models
```

## Upstream components

| Component | Repository | Tested revision | Role |
|---|---|---|---|
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | `f21265c1e2710b3bd2386596943f0007f55f802e` | Oobleck VAE and DiT initialization for Stage 2 |
| MuQ-MuLan-large | `OpenMuQ/MuQ-MuLan-large` | `2e01c796b71dca71b45251384c04cd7b237c9020` | Frozen music representation and Stage-2 condition encoder |
| Qwen3-VL-Embedding code | `QwenLM/Qwen3-VL-Embedding` | `8ce3aab1fbcc7b7143b094f0b2005dd89e7246b9` | Multimodal embedding implementation |
| Qwen3-VL-Embedding-2B | `Qwen/Qwen3-VL-Embedding-2B` | `2a50926d213628c727f38025982a76f655673f54` | Image, video, and text conditions for Stage 1 |

CLAP is used by parts of the paper evaluation harness as an FAD backbone.
It is not a dependency of the released Stage-2 inference path.

## Resolution order

`MericPipeline.from_pretrained` resolves each project checkpoint in this order:

1. A per-checkpoint environment override.
2. A matching file in `$MERIC_HOME`, which defaults to `~/.cache/meric`.
3. The pinned Hugging Face Hub revision.

| Component | Environment override |
|---|---|
| Stage-1 Music Head | `MERIC_RDM_CKPT` |
| Stage-2 flow decoder | `MERIC_STAGE2_CKPT` |
| Stable Audio Open snapshot | `MERIC_STABLE_AUDIO_DIR` |
| MuQ-MuLan snapshot | `MERIC_MUQ_DIR` |
| Meric Hub repository | `MERIC_HF_REPO` |
| Meric Hub revision | `MERIC_HF_REVISION` |
| Stable Audio repository | `MERIC_STABLE_AUDIO_REPO` |
| Stable Audio revision | `MERIC_STABLE_AUDIO_REVISION` |
| MuQ-MuLan repository | `MERIC_MUQ_REPO` |
| MuQ-MuLan revision | `MERIC_MUQ_REVISION` |

Repository overrides can point to mirrors or private staging repositories.
When a custom repository is selected without its matching revision override, Meric does not assume that the canonical release commit exists in the mirror.

Offline example:

```bash
export MERIC_HOME=/data/models/meric
export MERIC_STABLE_AUDIO_DIR=/data/models/stable-audio-open-1.0
export MERIC_MUQ_DIR=/data/models/MuQ-MuLan-large
export QWEN3VL_REPO=/data/repos/Qwen3-VL-Embedding

meric generate --image photo.jpg --output-dir outputs/offline
```

The expected files directly under `$MERIC_HOME` are:

```text
$MERIC_HOME/
├── rdm_sft_v3.pth
├── rdm_sft_instrumental.pth
└── mericldm.ckpt
```

## Manual prefetch

The normal path is automatic download on first use.
For an offline machine, prefetch the exact revisions on a connected machine:

```python
from huggingface_hub import hf_hub_download, snapshot_download

meric_revision = "0a199f0f6a0c7f64f397a2192882883bfb2c7736"
for filename in ("rdm_sft_v3.pth", "rdm_sft_instrumental.pth", "mericldm.ckpt"):
    hf_hub_download(
        repo_id="Audiofool/meric",
        filename=filename,
        revision=meric_revision,
    )

snapshot_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    revision="f21265c1e2710b3bd2386596943f0007f55f802e",
    allow_patterns=["vae/*", "transformer/*", "*.json"],
)

snapshot_download(
    repo_id="OpenMuQ/MuQ-MuLan-large",
    revision="2e01c796b71dca71b45251384c04cd7b237c9020",
)
```

The separate Qwen3-VL environment can be prepared with:

```bash
bash scripts/setup_qwen3vl.sh
```

The helper pins both the upstream code and model revisions listed above.
Set `QWEN3VL_REPO` to use an existing checkout and `QWEN3VL_MODEL` to use an existing model directory.

## Loading

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
```

The released checkpoints are loaded with `torch.load(weights_only=True)`.
The Stage-1 checkpoint is read once and reused for model type detection and state restoration.

## Checksum verification

Run the release verifier against a directory containing the three Meric files:

```bash
python scripts/release/verify_release.py --weights-dir /path/to/weights
```

The command checks file names, byte sizes, and SHA-256 digests against this release manifest.

## Licenses

The repository source is Apache-2.0.
The released Stage-2 checkpoint derives from Stable Audio Open and contains frozen MuQ-MuLan parameters.
It is governed by the applicable Stability AI Community License and CC BY-NC 4.0 terms.
Qwen3-VL and all other upstream assets retain their respective licenses.
See [`NOTICE`](../NOTICE) before redistribution or commercial use.
