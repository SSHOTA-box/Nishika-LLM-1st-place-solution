# License Scope

Copyright 2026 SShota

## Apache-2.0-licensed work

The independently authored implementation code and repository infrastructure are licensed under the [Apache License 2.0](LICENSE). This includes:

- `code/*.py`;
- `setup_dirs.py` and `setup_model_downloders.py`;
- `Dockerfile` and `docker-compose.yml`;
- `settings.json`, dependency manifests, tests, and CI configuration;
- operational documentation authored for using and contributing to the software, unless otherwise stated.
- the explicitly allowlisted generative-AI-created sample files at `data/train.csv`, `images/train/demo1/1.png`, and `images/train/demo2/1.png`, to the extent of SShota's rights in those files.

## Competition-specific archival material

`docs/COMPETITION.md`, `docs/COMPETITION_JA.md`, `docs/SOLUTION.md`, `docs/SOLUTION_JA.md`, and `docs/assets/Solution.png` record competition-specific analysis, results, and historical context. They are published under the archived competition policy's post-competition, non-commercial publication condition and are not granted under Apache-2.0.

## Excluded material

The Apache-2.0 license does not cover third-party dependencies or any ignored/non-distributed artifact, including:

- Qwen3-VL-8B-Instruct weights or upstream code;
- Python packages and container images;
- competition data, images, and metadata;
- the organizer-provided evaluation encoder;
- trained LoRA adapters, checkpoints, logs, predictions, and submissions;
- the original PDF/PPTX solution report;
- any non-allowlisted competition data or images stored locally under `data/` or `images/`;
- company names, competition names, logos, or trademarks.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream dependencies.
