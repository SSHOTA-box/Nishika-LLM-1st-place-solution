# Public Release Checklist

This file separates checks completed on the current working copy from decisions that require the maintainer or competition organizer.

## Completed in this working copy

- [x] Competition CSV/XLSX files, images, downloaded models, LoRA checkpoints, outputs, and submissions are ignored.
- [x] Restricted and large runtime artifacts are excluded from the Docker build context.
- [x] Only the explicitly allowlisted synthetic CSV and two synthetic images are tracked; remaining local data/image artifacts stay ignored.
- [x] The maintainer confirmed that the Competition Rules permit non-commercial publication of the solution analysis and trained model after the competition.
- [x] The trained LoRA adapter is not distributed through this repository; interested users are directed to contact SShota privately.
- [x] The original solution PDF/PPTX are ignored because they contain personal information.
- [x] The maintainer reports that the training and inference workflow has already been validated in the original environment.
- [x] Common credentials, local environment files, caches, and private keys are ignored.
- [x] Source/configuration files were scanned for common secret patterns, email addresses, private URLs, cloud paths, and absolute user paths; no credential was found.
- [x] Public commands use the real entry points under `code/`.
- [x] English and Japanese READMEs describe the same workflow.
- [x] Lightweight syntax, configuration, path, and setup checks are available in CI.

## Required before making the repository public

- [ ] Add a durable, non-restricted archive or citation for the relevant Competition Rules, non-commercial publication permission, and final result.
- [x] The maintainer confirmed authorship and ownership of the repository's original source code.
- [x] The maintainer confirmed that the implementation contains no competition-provided code.
- [x] Apache-2.0 was formally selected for the independently authored implementation code, with `SShota` as the public copyright-holder label.
- [x] The standard Apache-2.0 text, NOTICE, and a scope notice separate code from competition-specific analysis under the archived non-commercial publication condition.
- [ ] Confirm whether the organizer-provided text encoder may be used after the competition and document an authorized acquisition method, exact version, and checksum.
- [ ] Replace the competition/result TODOs with durable official URLs and independently verify rank and score.
- [ ] If publishing a solution report later, create and visually review a sanitized copy with names, affiliations, presenter details, and document metadata removed.
- [ ] Perform a visual/content review of all documents intended for publication; automated text scans cannot establish privacy or copyright clearance.
- [ ] Preserve a concise clean-clone reproduction log (environment versions, commands, and expected output paths) as public evidence of the completed validation.
- [ ] Pin the Qwen Hugging Face revision and record checksums for all external artifacts needed for exact reproduction.
- [x] Audited the reachable history and locally retained previous Git objects for common secrets, competition datasets, model files, and outputs; none were found in committed file contents.
- [ ] Review `git status --ignored` immediately before the first push and confirm that only intended source/documentation files are staged.

## Release and maintenance

- [ ] Publish an initial versioned release only after the reproducibility blockers above are resolved.
- [ ] Accept Issues for reproducible bugs and documentation gaps; request minimal examples without restricted data.
- [ ] Require CI and a description of GPU/data validation for pull requests.
- [ ] Review dependency updates in small batches and rerun the Docker smoke test.
- [ ] Keep `README.md` as the source of truth and update `README_JA.md` in the same pull request.
