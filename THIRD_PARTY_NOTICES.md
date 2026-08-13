# Third-Party Components

This repository's source code is maintained separately from the third-party software and model artifacts it uses. Those components are not relicensed by this repository.

## Qwen3-VL-8B-Instruct

- Upstream: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Declared upstream license: Apache License 2.0
- Repository treatment: model weights are not committed or redistributed; `setup_model_downloders.py` downloads them directly from the upstream model repository.

Users must review and comply with the license and model card attached to the exact upstream revision they download. The downloader currently does not pin a revision.

## Python and container dependencies

The Python packages listed in `requirements.txt` and `requirements_windows.txt`, and the `pytorch/pytorch` Docker base image, remain subject to their respective upstream licenses. They are installed or pulled during environment setup and are not covered by this repository's future code license.

## Excluded competition artifacts

The following are not third-party components distributed by this repository because they are excluded from Git:

- competition data and images;
- the organizer-provided evaluation encoder;
- downloaded model weights;
- trained LoRA adapters and checkpoints;
- generated predictions and submissions;
- original solution PDF/PPTX files containing personal information.

This notice is informational and is not a substitute for the license texts supplied by upstream projects.
