# Meric public release checklist

This is the maintainer gate for making Audiofool934/meric public.
It is separate from the public [roadmap](ROADMAP.md) because correctness, licensing, security, and factual consistency cannot be deferred as ordinary feature work.
Do not mark a check complete using evidence from an earlier commit.
Every required check must apply to the exact commit and artifact revisions selected for release.

## Release scope and Git history

- [ ] Choose the public version, release tag, and release date.
- [ ] Freeze the supported surface and confirm that every documented command belongs to it.
- [ ] Create a clean public root commit from the reviewed release tree while the repository is private.
- [ ] Confirm that public branches and tags expose only intended history.
- [ ] Keep gh-pages as the only additional public-history branch unless another branch is explicitly approved.
- [ ] Confirm that internal experiments, machine-specific paths, generated outputs, credentials, and private artifacts are absent from every public ref.
- [ ] Pass CI on the exact release commit before changing repository visibility.

## Factual and metadata consistency

- [ ] Confirm the canonical title, author order, contribution notes, venue, and year across README, CITATION, model card, and project page.
- [ ] If no arXiv record exists, describe it as forthcoming and provide no arXiv identifier or availability claim.
- [ ] If proceedings metadata is unassigned, omit pages and DOI rather than guessing.
- [ ] Remove placeholder authors, placeholder URLs, draft citations, and stale repository names.
- [ ] Check every public resource link and every relative documentation link.
- [ ] Confirm that README, ROADMAP, REPRODUCIBILITY, DATA_INVENTORY, and the model card describe the same artifact boundary.

## Model and source licensing

- [ ] Record the release decision for the MuQ-MuLan parameters serialized in the current Stage-2 checkpoint.
- [ ] If the embedded MuQ-MuLan state is retained, approve its redistribution boundary and keep the Stability AI Community License and CC BY-NC 4.0 disclosures visible.
- [ ] If the embedded state is removed, publish a new checkpoint revision and update every byte size, SHA-256 digest, model pin, and model card reference.
- [ ] Confirm every adapted-source attribution and license text in NOTICE.
- [ ] Confirm that the Apache-2.0 source license is never presented as the license for model weights or third-party datasets.

## Data release and reproducibility

- [ ] Approve the ARIA dataset card, immutable revision, license, provenance, and redistribution boundary before publishing any ARIA artifact.
- [ ] Confirm that every published manifest and embedding is permitted for redistribution.
- [ ] Validate released manifests for stable identifiers, split isolation, deterministic ordering, expected counts, and checksums.
- [ ] Validate released arrays for exact dimensions, float32 type, and finite values.
- [ ] Record preprocessing commands, model revisions, normalization, and instructions for every published embedding.
- [ ] Run a cold pretrained-inference reproduction from a clean environment and retain its generated manifest.
- [ ] If the release claims full paper reproduction, complete every data and evaluation item in ROADMAP first.
- [ ] Otherwise, state the partial reproduction boundary without ambiguity on every public surface.

## Hugging Face and project page

- [ ] Upload the reviewed model card without re-uploading checkpoints unless their digests changed.
- [ ] Verify the returned immutable Hugging Face revision and all three remote checkpoint digests.
- [ ] Confirm that the model revision pinned in meric/hub.py matches the reviewed Hub revision.
- [ ] Remove the premature arXiv availability claim from the live project page until the record exists.
- [ ] Confirm that GitHub, documentation, model-weight, citation, and license links resolve from the live project page.
- [ ] Confirm that the live Demo remains served from the OSS repository's gh-pages branch.

## Security and community

- [ ] Choose a monitored security contact or enable GitHub private vulnerability reporting immediately after the repository becomes public.
- [ ] Verify that SECURITY contains a reporting route that is actually available.
- [ ] Choose a community contact and publish a code-of-conduct policy or an explicit contribution-behavior policy.
- [ ] Confirm that issue and pull-request guidance do not request private datasets, credentials, or copyrighted media.

## Final verification

- [ ] Run Ruff lint over meric, scripts, and tests.
- [ ] Run Ruff format checking over meric, scripts, and tests.
- [ ] Run the complete CPU test suite.
- [ ] Compile all Python sources under meric and scripts.
- [ ] Build the wheel and source distribution.
- [ ] Run Twine checks over every built distribution.
- [ ] Run scripts/release/verify_release.py.
- [ ] Verify all release checkpoint sizes and SHA-256 digests with verify_release.py --weights-dir.
- [ ] Scan the exact public tree and intended Git history for credentials.
- [ ] Install the built wheel in a clean environment and run import, CLI, and model-list smoke tests.
- [ ] Confirm the release worktree and index contain no unintended changes.

## Launch and post-launch

- [ ] Push the reviewed clean history while the repository is still private.
- [ ] Create and verify the signed or annotated release tag.
- [ ] Change visibility only after every non-conditional gate above is complete.
- [ ] Enable and test the selected private security-reporting route.
- [ ] Verify a fresh anonymous clone, documentation links, Hugging Face downloads, and the live Demo.
- [ ] Publish release notes that state the supported surface and link this roadmap.
- [ ] Archive the final test output, artifact revisions, and release commit for future audits.
