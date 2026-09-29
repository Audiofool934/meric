# Meric release roadmap

This roadmap distinguishes artifacts that are available now from work that is planned or waiting on external records.
Checked items are present in this repository or in a linked artifact at a pinned revision.
Unchecked items are not currently available and must not be described as released elsewhere.

## Available now

- [x] Image-, video-, text-, and MuQ-conditioned 44.1 kHz music inference.
- [x] Pinned Meric model resolution with exact checkpoint byte sizes and SHA-256 digests.
- [x] Deterministic seeds and per-run manifests containing settings, revisions, and file checksums.
- [x] Music Head pretraining and fine-tuning code with resumable checkpoints.
- [x] Exact camera-ready result tables and a documented public artifact boundary.
- [x] CPU tests, package builds, source checks, and an offline release verifier.

## Publication records

- [ ] Add the arXiv identifier and immutable URL after the record is assigned.
- [ ] Add the official ECCV proceedings pages and DOI after publication.
- [ ] Add author ORCID identifiers if the authors choose to publish them.

## Data and training artifacts

- [ ] Publish an ARIA dataset card with an immutable revision, source provenance, license, redistribution decision, and the 4K/1K split.
- [ ] Publish permitted train and test JSONL manifests with stable identifiers, sample counts, and SHA-256 checksums.
- [ ] Publish permitted prepared embeddings or deterministic generation recipes for every condition and target embedding.
- [ ] Add a dataset artifact verifier for split integrity, dimensions, finite values, path resolution, and checksums.
- [ ] Publish the permitted Stage-2 training manifest and a supported end-to-end trainer after the data access and redistribution boundary is approved.

## Evaluation artifacts

- [ ] Publish baseline environment locks, immutable model revisions, and generated-output manifests.
- [ ] Publish metric-extractor revisions and per-sample evaluation manifests.
- [ ] Publish semantic-judge prompts, raw machine outputs, anonymized human annotations, and the analysis recipe.
- [ ] Publish training hardware, wall-clock time, peak memory, and final checkpoint-selection records.

## Packaging and usability

- [ ] Evaluate smaller inference-only Music Head artifacts alongside the full resumable checkpoints.
- [ ] Add a public release milestone and link completed roadmap items to their immutable release artifacts.

## Scope

The current supported surface is pinned pretrained inference, Music Head training-code inspection and execution with prepared inputs, and partial Flow Decoder training-code inspection.
Exact paper retraining and full table regeneration are not claimed until the required data and evaluation artifacts above are available.
An unchecked item has no promised release date unless a dated GitHub milestone says otherwise.

Maintainers use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for correctness, licensing, security, and launch gates that cannot be deferred as ordinary roadmap work.
