[English](README.md) | [日本語](README_JA.md)

# Revamp × Nishika LLM Competition — 1st Place Solution

Training, inference, and ensemble code used by team SSS_lab.

The competition was jointly organized by **Revamp Corporation** and **Nishika Inc.** The task was to generate natural and persuasive Japanese promotional copy from product information such as product images and product names. The generated text was expected to communicate product appeal and encourage purchase while remaining consistent with the provided information, rather than simply listing specifications.

The code in this repository directly uses a product ID and its corresponding one or two product images as input.

## Result

| Item | Value |
| --- | --- |
| Competition | Student-only Revamp × Nishika LLM Competition: Product PR Text Generation for a Major Global Retailer |
| Rank | 1st / 48 teams |
| Team | SSS_lab |
| Final leaderboard score | `0.639706` |
| Task | Generate Japanese product PR text of at most 100 characters from one or two images |
| Metric | Mean cosine similarity from the organizer-provided text encoder |

See [Competition Information](docs/COMPETITION.md) for the competition dates and publication policy.

- [Official competition report by Revamp Corporation](https://revamp.co.jp/interview/13/)

## Solution

![Solution diagram combining Qwen3-VL, LoRA, the official Sentence-BERT encoder, and embedding-centroid selection](docs/assets/Solution.png)

1. Load Qwen3-VL-8B-Instruct in 4-bit NF4.
2. Fine-tune it for product PR generation with LoRA.
3. Generate 100 candidates at temperature 0.9 and 100 at 1.2.
4. Embed the candidates with the organizer-provided text encoder.
5. Select the text nearest to the centroid of all 200 candidates.
6. Format the output to at most 100 characters.

See [Detailed Solution](docs/SOLUTION.md) for comparative experiments and complete settings.

## Repository Structure

```text
.
├── code/
│   ├── train.py              # LoRA training
│   ├── predict_v1.py         # Temperature 0.9, 100 candidates
│   ├── predict_v2.py         # Temperature 1.2, 100 candidates
│   └── marge.py              # Merge 200 candidates and create final output
├── data/
│   ├── train.csv             # ID,label (fictional sample included)
│   ├── test.csv              # ID (not included)
│   └── sample_submission.csv # ID,target (not included; unused by the code)
├── images/
│   ├── train/<ID>/*.{jpg,jpeg,png,bmp} # two fictional samples included
│   └── test/<ID>/*.{jpg,jpeg,png,bmp}  # not included
├── models/                   # Base model, encoder, and LoRA
├── outputs/                  # Inference outputs
├── docs/                     # Competition and solution documentation
├── tests/
├── settings.json
├── Dockerfile
└── docker-compose.yml
```

The original competition data and images are not included. The bundled CSV and images are fictional layout examples, not training data.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit

The validated environment used an NVIDIA RTX 6000 Ada with 48 GB VRAM and 64 GB system RAM.

## Setup

Run commands from the repository root.

```bash
docker compose build
docker compose run --rm nishika-env python setup_dirs.py
docker compose run --rm nishika-env python setup_model_downloders.py
```

`setup_model_downloders.py` downloads Qwen3-VL-8B-Instruct to `models/Qwen3-VL-8B-Instruct/`.

The organizer-provided evaluation encoder is not redistributed, and no public download source is configured. Users with legitimate access can place the complete Hugging Face model directory at:

```text
models/predicter_LLM/
```

For a functional test, it can be replaced with a general text encoder loadable through Transformers `AutoTokenizer` and `AutoModel`. This replacement will not reproduce scores from the organizer-provided encoder.

## Models

| Model | Path | Availability |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `models/Qwen3-VL-8B-Instruct/` | Downloaded by setup script |
| Evaluation encoder | `models/predicter_LLM/` | Organizer version not distributed; a general encoder can substitute |
| Trained LoRA | `models/LoRA/LoRA_Qwen3-VL-8B-Instruct/` | Not distributed here |

The trained LoRA adapter is not distributed through this repository; interested users may contact SShota privately.

## Training

```bash
docker compose run --rm nishika-env python code/train.py
```

Main settings are seed 42, the last 300 rows for validation, effective batch size 8, and learning rate `1e-4`. The current code is configured for four epochs.

## Inference

Set `LORA_ADAPTER` in `settings.json` to the checkpoint to use. The competition configuration used `checkpoint-1599`.

```bash
docker compose run --rm nishika-env python code/predict_v1.py
docker compose run --rm nishika-env python code/predict_v2.py
docker compose run --rm nishika-env python code/marge.py
```

The final file is written to:

```text
outputs/LoRA_inference/final_submission/final_submission_ensemble_200.csv
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Publication Scope

Included:

- Training, inference, and ensemble code
- Docker environment and configuration
- Solution description and comparative experiments
- Fictional sample data and images

Not included:

- Original competition data and images
- Organizer-provided text encoder
- Trained LoRA adapter
- Generated predictions, submissions, and logs
- Original PDF and PowerPoint reports containing personal information

Questions are welcome. Please get in touch if any part of the publication scope or solution interests you or requires clarification.

## License

The implementation code was independently written by SShota and is released under the [Apache License 2.0](LICENSE).

Competition-specific solution documents are published under the archived Competition Rules' post-competition, non-commercial publication condition. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the exact scope and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external dependencies.

## Disclaimer

This is not an official repository of Revamp Corporation or Nishika Inc. Company names, competition names, and trademarks belong to their respective owners.
