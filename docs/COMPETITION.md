[English](COMPETITION.md) | [日本語](COMPETITION_JA.md)

> License note: this competition-specific archival document is not licensed under Apache-2.0. It is published under the archived competition policy's post-competition, non-commercial publication condition. See [LICENSE_SCOPE.md](../LICENSE_SCOPE.md).

# Competition Information

## Overview

Team SSS_lab entered the following competition and placed first among 48 teams. This repository publishes and preserves the solution developed for the competition.

**Student-only Revamp × Nishika LLM Competition: Product PR Text Generation for a Major Global Retailer**

The competition was jointly organized by **Revamp Corporation** and **Nishika Inc.** Participants generated compelling Japanese promotional copy from product information, including product images and product names. The objective was not merely to list specifications, but to communicate product appeal naturally and persuasively while remaining consistent with the supplied information.

The checked-in pipeline consumes the product ID and its associated images. Product-name availability belongs to the broader competition description and should not be read as an additional input used by every script in this repository.

## Competition Summary

| Item | Value |
| --- | --- |
| Platform | Nishika |
| Organizers | Revamp Corporation × Nishika Inc. |
| Task | Multimodal / LLM-based product PR text generation |
| Eligibility | University and graduate students |
| Start | December 1, 2025 |
| Submission deadline | January 14, 2026 |
| Participants | 98 |
| Submissions | 664 |

## Final Result

- **Rank:** 1st place
- **Team:** SSS_lab
- **Final score:** `0.639706`

This repository documents the solution used for the final submission. Individual members' names and affiliations are not published.

## Evaluation

Generated PR text was evaluated against reference PR text using an organizer-adjusted embedding model:

1. Convert the generated and reference texts into embeddings.
2. Calculate cosine similarity for each sample.
3. Average the sample-level cosine similarities.

Competition materials identify `sentence-bert-base-ja-mean-tokens` as the base of the organizer's custom evaluation model. The organizer fine-tuned this model on the competition data. The test evaluation was split approximately 50/50 between the provisional and final leaderboards, with the final split determining the final rank.

The organizer-adjusted encoder is a competition-provided artifact. It is required by the checked-in training evaluation, candidate selection, and ensemble code, but it is not distributed by this repository.

## Publication Policy

The archived competition policy permitted publication of the following after the competition for **non-commercial purposes**:

- models;
- analysis results.

The listed publication channels included social media, blogs, source-code repositories, and academic papers/citations. The technical analysis is published under that permission. The trained LoRA adapter is not distributed through this repository; interested users may contact SShota privately.

This permission is not treated as permission to redistribute the original competition dataset, images, metadata, organizer encoder, or generated outputs derived from restricted data. It also does not automatically establish an open-source license for the repository's original code.

## Data Notice

This repository intentionally excludes:

- original training and test data;
- competition-provided images and metadata;
- the organizer-provided evaluation encoder;
- other competition files whose redistribution rights are unclear;
- generated outputs that reproduce or derive from restricted competition content.

## Nishika Service Closure

Nishika announced that its competition, recruitment, and related services would end on **March 31, 2026**. Affected competition functions included participation, submissions, ranking access, and discussions. The official announcement is preserved at [Nishika's service-closure notice](https://info.nishika.com/news/close).

Because the original competition service is no longer a reliable long-term source, this repository also preserves the competition context and winning solution.

## Archival Purpose

The repository preserves:

- the competition and task;
- the evaluation method and final result;
- the winning solution methodology;
- training and inference procedures;
- publication and redistribution restrictions;
- historical context following the platform shutdown.

The goal is to keep the solution understandable and reproducible to the extent permitted, even if the original competition pages are unavailable.

## Sources and Provenance

- [Official competition report by Revamp Corporation](https://revamp.co.jp/interview/13/)
- [Nishika service-closure notice](https://info.nishika.com/news/close)
- SSS_lab solution report dated February 13, 2026
- Retained Competition Rules and leaderboard records

Revamp's official report confirms the joint organization, competition period, task, approximately 100 participants, more than 660 submissions, and the presentation by the first-place entrant. The team name, exact final score, and detailed comparative experiments are based on the retained leaderboard records and solution report.

## Disclaimer

This is an independent solution publication and technical archive. It is not an official repository of Revamp Corporation or Nishika Inc. Company names, competition names, and trademarks belong to their respective owners.
