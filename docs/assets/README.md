# README figures

`method.png` is the unmodified framework figure, `method-meric-v5.png`, from the submitted ECCV 2026 camera-ready paper.
The figure's Stage I and Stage II labels describe training order: acoustic decoder training precedes Music Head training.

The input images and generated mel spectrograms under `examples/` are copied without alteration from the same paper's qualitative examples.
The renamed files correspond to these original sample identifiers:

| README example | Original sample ID | Generated spectrogram |
|---|---|---|
| Guitar ensemble | `5zkBdG-8FfQ` | `5zkBdG-8FfQ_ours.png` |
| Pop art | `57726e4aedc2cb3880b69f33` | `57726e4aedc2cb3880b69f33_ours.png` |
| Mountain landscape | `5GeT5PLLAhw` | `5GeT5PLLAhw_ours.png` |

These are research examples from the paper, not additional released training data.
Source images retain their upstream terms; see [NOTICE](../../NOTICE).
The music panels show generated mel spectrograms for 10-second, 44.1 kHz outputs.
Audio playback is available on the [project page](https://audiofool.blog/meric/).

`generation-results.png` plots FAD-CLAP on MelBench and MusicCaps directly from [the camera-ready result tables](../RESULTS.md).
Each panel includes every baseline in its source table, uses a linear axis starting at zero, and reports exact values.
The panels use different axis ranges to keep their values legible.
The figure is a selected benchmark summary; other datasets and metrics remain in the complete tables.

To regenerate the chart from the repository root:

```bash
uv run --no-project --with matplotlib==3.11.2 python docs/assets/render_results.py
```
