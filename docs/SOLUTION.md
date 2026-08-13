[English](SOLUTION.md) | [日本語](SOLUTION_JA.md)

# Solution and Comparative Experiments

This document describes the first-place solution and comparative experiments used in the Revamp × Nishika LLM competition. See [COMPETITION.md](COMPETITION.md) for the competition background, metric, and publication scope, and [README.md](../README.md) for usage instructions.

> [!NOTE]
> This is competition-specific analytical material. It is not covered by Apache-2.0 and is published under the archived Competition Rules' post-competition, non-commercial publication condition. See [LICENSE_SCOPE.md](../LICENSE_SCOPE.md).

## Solution Summary

![Solution diagram combining Qwen3-VL, LoRA, the official Sentence-BERT encoder, and embedding-centroid selection](assets/Solution.png)

The final solution separates generation from selection:

1. Load Qwen3-VL-8B-Instruct in 4-bit NF4.
2. Adapt the language and visual-merger layers to product-copy generation with LoRA.
3. Generate 100 candidates at temperature 0.9 and another 100 at 1.2 for each product.
4. Embed the candidates with the organizer-provided evaluation encoder.
5. Compute the centroid of the normalized candidate embeddings.
6. Select the generated candidate nearest to that centroid.
7. Remove line breaks and surrounding quotation marks, then truncate to 100 characters.

The method selects a representative from actual generated texts; it does not attempt to decode the mean embedding into text.

## Training Configuration

The current `code/train.py` uses the following main settings.

| Item | Setting |
| --- | --- |
| Base model | Qwen3-VL-8B-Instruct |
| Quantization | 4-bit NF4 |
| Validation | Last 300 CSV rows |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Epochs | 4 |
| Learning rate | `1e-4` |
| LoRA dropout | 0.05 |
| Maximum sequence length | 2048 |
| Precision | bfloat16 |
| Seed | 42 |

The LoRA rank and alpha vary by layer role.

| Target | Rank | Alpha |
| --- | ---: | ---: |
| Visual deepstack / merger | 16 | 32 |
| Attention (`q/k/v/o_proj`) | 32 | 64 |
| FFN (`gate/up/down_proj`) | 64 | 128 |

The post-competition solution report records three training epochs, while the code currently published here is configured for four. The competition submission used `checkpoint-1599`, corresponding to the three-epoch run. Reproduction of the reported score with the final adapter from the current four-epoch configuration has not been verified.

## Inference Configuration

| Item | v1 | v2 |
| --- | ---: | ---: |
| Candidates | 100 | 100 |
| Temperature | 0.9 | 1.2 |
| `top_p` | 0.95 | 0.95 |
| `top_k` | 50 | 50 |
| `max_new_tokens` | 200 | 200 |

`code/marge.py` combines both candidate sets and selects the text nearest to the centroid of all 200 candidates.

## Comparative Experiments

The following experiments were run during or after the competition. They were not rerun while preparing this repository. Leaderboard results are identified separately from the post-competition validation experiment.

### Experiment 1: Temperature and Ensemble

| Generation | Candidates | Provisional leaderboard | Final leaderboard |
| --- | ---: | ---: | ---: |
| Temperature 0.9 | 100 | 0.6407 | 0.6321 |
| Temperature 1.2 | 100 | 0.6469 | 0.6374 |
| **0.9 + 1.2 ensemble** | **200** | **0.6476** | **0.6397** |

The two-temperature ensemble corresponds to the official final score of `0.639706`. Temperature 1.2 outperformed 0.9, and combining both improved the score further. The combined pool contains both relatively stable outputs and more diverse candidates.

### Experiment 2: Candidate Count K

Temperature was fixed at 0.9.

| K | Provisional leaderboard |
| ---: | ---: |
| 1 | 0.5641 |
| 5 | 0.5917 |
| 10 | 0.6137 |
| 50 | 0.6217 |
| **100** | **0.6407** |

The improvement from K=1 to K=100 was `+0.0766`. More candidates may cover the generation space more broadly and allow the centroid to estimate shared semantics more reliably. K greater than 100 was not tested, so these results do not establish indefinite improvement. Generating K=100 for all 1,190 test products took roughly two days in the execution environment.

### Experiment 3: Vision-Language Model

These models were compared without LoRA or competition-data fine-tuning.

| Model | Train score | Provisional leaderboard |
| --- | ---: | ---: |
| Qwen2.5-VL-7B-Instruct | 0.511 | 0.5021 |
| Qwen3-VL-4B-Instruct | 0.509 | 0.4980 |
| **Qwen3-VL-8B-Instruct** | **0.515** | **0.5051** |

Qwen3-VL-8B-Instruct was best among the compared models, but the differences were small. Candidate count and encoder selection produced larger recorded changes, suggesting that domain adaptation and inference-time search mattered in addition to base-model choice.

### Experiment 4: Embedding Model for Candidate Selection

Only the candidate-selection encoder was changed at K=100.

| Embedding model | Provisional leaderboard |
| --- | ---: |
| `sonoisa/sentence-bert-base-ja-mean-tokens` | 0.6120 |
| **Organizer-provided evaluation encoder** | **0.6407** |

The difference was `+0.0287`. Using the same semantic space for candidate selection and evaluation aligns the selection objective with the competition metric. The organizer fine-tuned this evaluation encoder on the competition data. It is not distributed in this repository.

### Experiment 5: Distance from the Centroid

This post-competition experiment used the last 300 training items as validation data. Zero percent denotes the nearest candidate and 100% the farthest.

| Distance rank from centroid | Validation score |
| --- | ---: |
| Oracle | **0.7828** |
| **0% (nearest)** | **0.6167** |
| 10% | 0.5955 |
| 20% | 0.5777 |
| 30% | 0.5645 |
| 40% | 0.5570 |
| 50% | 0.5407 |
| 60% | 0.5335 |
| 70% | 0.5050 |
| 80% | 0.4728 |
| 90% | 0.4289 |
| **100% (farthest)** | **0.2245** |

The score decreases almost monotonically with distance, supporting the hypothesis that candidates near the centroid capture more stable semantics in the generation distribution.

The oracle assumes access to the ground-truth score for every generated candidate and is not an available inference method. Its gap from the nearest-centroid result (`0.6167`) suggests that candidate reranking still had substantial headroom.

## Interpretation

The solution avoids relying on a single stochastic generation. Averaging many candidate embeddings may preserve semantics that recur across candidates while reducing the relative influence of individual hallucinations and outliers.

Using the evaluation encoder for selection also makes the process metric-aligned: selection occurs in the same semantic space used for scoring, rather than only optimizing general Japanese semantic similarity. The method can be summarized as:

> Generate many candidates, embed them in the competition metric space, and choose the candidate nearest to the centroid.

## Limitations and Future Work

- These are single-condition results without multi-seed averages or confidence intervals.
- Leaderboard comparisons may not isolate every variable perfectly and should not be interpreted as strict causal ablations.
- The official score cannot be fully reproduced from this repository alone because the organizer encoder and original competition data are not distributed.
- The oracle gap motivates learned reranking, image-text consistency checks, OCR-based factual verification, and cluster or medoid selection, but these are untested future directions.
- CUDA sampling and training are not guaranteed to be bitwise deterministic.

## Provenance

- SSS_lab solution report dated 2026-02-13
- `code/train.py`, `code/predict_v1.py`, `code/predict_v2.py`, and `code/marge.py` in this repository
- Retained Competition Rules, leaderboard records, and competition information

This is an independent technical archive, not an official document of Revamp Corporation or Nishika Inc.
