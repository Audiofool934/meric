# Sample images

These procedurally generated images are synthetic samples created for this repository.
They are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

| File | Description |
|---|---|
| `sample_sunset.jpg` | Warm radial glow over a dusk gradient |
| `sample_ocean.jpg` | Cool teal and blue wave bands |
| `sample_abstract.jpg` | Colorful interfering sinusoids |

Run an image-conditioned sample from the repository root:

```bash
meric generate \
  --image examples/images/sample_sunset.jpg \
  --output-dir outputs/sunset \
  --seed 42
```

Any readable JPEG or PNG can be used instead.
