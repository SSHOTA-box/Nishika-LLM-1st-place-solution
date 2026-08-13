# Model artifacts

Downloaded weights, trained adapters, and checkpoints are not stored in Git.

Expected layout:

```text
models/
├── Qwen3-VL-8B-Instruct/             # public base model
├── predicter_LLM/                     # organizer-provided encoder
└── LoRA/
    └── LoRA_Qwen3-VL-8B-Instruct/     # training outputs/checkpoints
```

Run `python setup_model_downloders.py` to download the Qwen base model. The organizer encoder is not redistributed and has no public download source configured here. Only users with legitimate access should place it under `predicter_LLM/`.

For a functional test, `predicter_LLM/` may instead contain a general text encoder loadable through Transformers `AutoTokenizer` and `AutoModel`. A substitute encoder will not reproduce scores from the organizer-provided encoder.

The trained LoRA adapter is not distributed through this repository. Its publication is permitted under the competition policy, so interested users may contact SShota privately. Keep all adapters, checkpoints, optimizer state, logs, the organizer encoder, and downloaded base-model weights out of Git.
