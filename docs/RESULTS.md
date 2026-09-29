# Camera-ready results

This document transcribes the quantitative tables in the final ECCV 2026 camera-ready source.
It is the public result reference for this repository.
Values in the extended internal experiment registry are not substituted for camera-ready values.

The paper evaluates generation with Frechet Audio Distance in MERT, CLAP, and VGGish feature spaces, KL divergence in MERT space, and an LLM-based semantic score named Sem.
Lower FAD and KL are better.
Higher Sem and retrieval recall are better.

## Vision-to-music generation

| Method | MelBench FAD-MERT | FAD-CLAP | FAD-VGG | KL-MERT | Sem |
|---|---:|---:|---:|---:|---:|
| Art2Mus | 31.60 | 0.945 | 13.72 | 0.0145 | 39.1 |
| GVMGen | 3.55 | 0.483 | 5.78 | 0.0072 | 52.9 |
| AudioX | 16.48 | 0.723 | 13.69 | 0.0127 | 53.8 |
| Qwen3-VL + MusicGen | 4.42 | 0.497 | 6.56 | 0.0076 | **80.6** |
| Qwen3-VL + AudioLDM2-L | 6.57 | 0.408 | 5.92 | 0.0078 | 77.6 |
| **Meric** | **1.55** | **0.127** | **1.27** | **0.0059** | 74.7 |

| Method | MuImage FAD-MERT | FAD-CLAP | FAD-VGG | KL-MERT | Sem |
|---|---:|---:|---:|---:|---:|
| Art2Mus | 30.75 | 0.902 | 15.28 | 0.0151 | 45.3 |
| GVMGen | 3.78 | 0.549 | 8.73 | 0.0086 | 49.6 |
| AudioX | 13.90 | 0.623 | 14.44 | 0.0130 | 51.8 |
| Qwen3-VL + MusicGen | 2.99 | 0.418 | 8.93 | 0.0079 | 85.6 |
| Qwen3-VL + AudioLDM2-L | 4.59 | 0.274 | 6.08 | 0.0079 | **87.2** |
| **Meric** | **1.79** | **0.169** | **3.20** | **0.0078** | 87.1 |

| Method | EmoMV FAD-MERT | FAD-CLAP | FAD-VGG | KL-MERT | Sem |
|---|---:|---:|---:|---:|---:|
| Art2Mus | 33.86 | 1.109 | 14.55 | 0.0147 | 44.3 |
| GVMGen | 3.96 | 0.617 | 7.20 | **0.0057** | 63.6 |
| AudioX | 4.75 | 0.487 | 5.45 | 0.0071 | 79.0 |
| Qwen3-VL + MusicGen | 7.08 | 0.736 | 7.65 | 0.0083 | 80.3 |
| Qwen3-VL + AudioLDM2-L | 8.61 | 0.609 | 7.41 | 0.0086 | 78.7 |
| **Meric** | **3.06** | **0.253** | **2.45** | 0.0060 | **81.0** |

| Method | ARIA FAD-MERT | FAD-CLAP | FAD-VGG | KL-MERT | Sem |
|---|---:|---:|---:|---:|---:|
| Art2Mus | 15.05 | 0.560 | 7.02 | 0.0077 | 45.3 |
| GVMGen | 4.18 | 0.337 | 5.83 | **0.0052** | 60.0 |
| AudioX | 20.15 | 0.766 | 11.71 | 0.0120 | 38.5 |
| Qwen3-VL + MusicGen | 5.09 | 0.283 | **2.04** | 0.0059 | 76.6 |
| Qwen3-VL + AudioLDM2-L | 9.95 | 0.262 | 3.66 | 0.0069 | 77.6 |
| **Meric** | **3.30** | **0.247** | 2.08 | 0.0055 | **79.6** |

## Text-to-music generation

| Method | MusicCaps FAD-MERT | FAD-CLAP | Sem | ARIA FAD-MERT | FAD-CLAP | Sem |
|---|---:|---:|---:|---:|---:|---:|
| MusicGen-M | 3.80 | 0.421 | 65.1 | 3.00 | 0.272 | 71.1 |
| MusicGen-L | 3.51 | 0.409 | 67.1 | 2.55 | 0.248 | 72.4 |
| AudioLDM2-Music | 4.60 | 0.260 | 68.0 | 4.63 | 0.251 | 68.6 |
| AudioLDM2-Large | 4.17 | 0.178 | **76.7** | 3.93 | 0.211 | **77.5** |
| Stable Audio Open | 3.85 | 0.268 | 68.3 | 3.50 | 0.324 | 57.8 |
| **Meric** | **1.87** | **0.110** | 74.2 | **2.14** | **0.155** | 74.5 |

## Vision-music retrieval

The table reports recall percentages in both directions.

| Method | MelBench I-to-M R@1 | R@10 | M-to-I R@1 | R@10 | ARIA I-to-M R@1 | R@10 | M-to-I R@1 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageBind | **4.38** | **18.00** | **3.66** | **17.00** | 0.30 | 2.30 | 0.13 | 1.77 |
| **Meric** | 1.34 | 8.62 | 1.12 | 9.54 | **0.40** | **2.80** | **0.39** | **2.40** |

## Text-music retrieval

| Method | MusicCaps T-to-M R@1 | R@10 | M-to-T R@1 | R@10 | ARIA T-to-M R@1 | R@10 | M-to-T R@1 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLAP | **4.31** | **19.48** | **5.54** | **23.53** | 0.17 | 2.30 | 0.63 | 3.73 |
| **Meric** | 3.27 | 18.80 | 2.82 | 16.42 | **0.83** | **5.47** | **0.73** | **6.40** |

## Music Head projector ablation

This ablation uses Qwen3-VL-Embedding features and reports the Music Head without ensembling.

| Method | Loss | MelBench FAD-MERT | ARIA FAD-MERT | MelBench R@1 / R@10 | ARIA R@1 / R@10 |
|---|---|---:|---:|---:|---:|
| Linear | Contrastive | 12.94 | 20.04 | **1.70** / 9.02 | 0.70 / **5.50** |
| Linear | MSE | 4.37 | 7.01 | 0.82 / 5.56 | 0.50 / 4.10 |
| BridgeVAE | MSE + KL | 4.06 | 9.35 | 0.84 / 6.28 | **0.80** / 3.50 |
| CrossVAE | MSE + KL + cosine | 3.48 | 7.22 | 0.98 / 5.88 | **0.80** / 4.30 |
| **Music Head** | Diffusion | **1.96** | **4.32** | 1.52 / **9.30** | **0.80** / 4.80 |

## Vision encoder ablation

| Backbone | Dimension | MelBench FAD-MERT | ARIA FAD-MERT | MelBench R@1 / R@10 | ARIA R@1 / R@10 |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-H-14 | 1024 | 2.54 | 4.73 | **1.60 / 10.00** | **0.90** / 4.10 |
| **Qwen3-VL-Embedding** | 2048 | **1.95** | **3.98** | 1.52 / 9.30 | 0.80 / **4.80** |

## Dataset summary

Training sets are deduplicated by YouTube ID.
MuImage training items are subsumed by AudioSet-Music after deduplication.

| Dataset | Input | Train | Test | Usage |
|---|---|---:|---:|---|
| AudioSet-Music | video | 359K | 4,660 | Pretraining |
| MelBench | image | 5,068 | 5,000 | Pretraining, generation, retrieval |
| EmoMV | video | 4,501 | 1,161 | Pretraining, generation |
| MuImage | image | subsumed | 2,536 | Generation |
| MusicCaps | text | not used | 2,745 | Generation, retrieval |
| ARIA | image | 4,000 | 1,000 | Fine-tuning, generation, retrieval |

## Semantic-judge validation benchmark

The camera-ready evaluation uses 300 six-way image-to-audio multiple-choice tasks: 100 MelBench, 100 Unsplash, and 100 WikiArt.
Each task contains one matching audio and five distractors.
Random top-1 and top-3 accuracy are 16.7 percent and 50.0 percent.

The submitted figure reports an overall top-1 accuracy of 44.0 percent for Gemini 3 Flash.
The complete raw human annotations and machine-judge outputs are not yet part of the public artifact set, so this result cannot yet be independently regenerated from the repository alone.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact public reproducibility boundary.
