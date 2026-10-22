# Reproduction notes

## Pinned components

| Component | Recorded version |
|---|---|
| Official FLUX source | `830f1ce178d7aea52f18fde31d8e8e36ce6bae21` |
| RoboDojo | `09bd5f5e762b949f5858ece31347ce62057b3a5e` |
| XPolicyLab | `f6e309e8ece20bfa33c0e5d7b05a5f3287522757` |
| Dataset repository | `RoboDojo-Benchmark/RoboDojo`, revision `35efbc7dedfdbeeb6e95fb749bd885d73d483e41` |
| Policy Python / Torch | Python 3.12, Torch 2.10 CUDA 12.8 |
| Policy libraries | official LeRobot 0.6.1; unpatched Transformers 5.16.1 |
| Simulator | separate Python 3.11 environment, Isaac Sim 5.1 |

Simulator dependency metadata compatibility handling is preserved in the archived installer sources. Do not call it an entirely unmodified installation. Static rendering, some interface checks and older-task tests passed, but these do not certify a newly rebuilt environment.

## Data and model inputs

- Cover: official episodes 300–399; charger: 2200–2299. Shared video/parquet chunks can contain other tasks: filter episode IDs explicitly.
- Source RGB video PTS, action/state rows and task instructions must agree. A few source camera spans were one frame shorter than numeric episode lengths; the final official-history index restricts complete windows to the common half-open source boundary, never the next episode.
- Preserve the official native `WindowDataset` selection/order for the baseline. Do not reuse the historical padded-tail/custom-mask dataset or custom prefix-memory installer.
- Compute normalization only from the selected training demonstrations. Do not reuse another task's quantiles or obtain cues from evaluation state.
- Only RGB, measured joints and literal instructions enter the policy. Evaluation IDs used in logs are not policy observations.

## Portable paths and historical safety checks

Archived paths use `/workspace/hdd/codex_remote_ops/…` and `/workspace/ssd/codex_remote_ops/…`; platform job/resource IDs are placeholders. Set up your own storage and resource allocation. Several files are historical point-in-time audit, repair or cleanup utilities, not a sequential installation recipe. In particular, do not run every shell script in the directory.

Original remote SHA gates refer to original bytes and are **not valid after path redaction or redeployment**. Regenerate gates from your own validated sources/data/configuration; do not simply remove assertions. `docs/SOURCE_MANIFEST.json` records original and public-source hashes for the recovered files.

The portable config template in `configs/` is reconstructed from the recorded recipe. It is not a byte-identical original remote config. Normalization is deliberately left to the verified index statistics. The source generators preserve the implementation history; several require previous-stage receipts that are not shipped here.

## Evaluation scope

Use the `r2_eval150` family, not the earlier single-seed `r2_final50` wrappers. Freeze the model and protocol before testing. Official environment configuration is assembled from the pinned official `main()` configuration AST block without launching the simulator during CPU checks; the final runtime must still verify the frozen configuration and execute actual episodes.

Each evaluation seed means its official layout-bank directory (0, 1 or 2). The exported policy inference seed remains fixed. Success rate and official Score are distinct metrics. Population standard deviation (`ddof=0`) is the declared primary cross-run summary; sample standard deviation is also recorded.

Infrastructure failures halt the wrapper and preserve valid prefixes rather than silently choosing replacement layouts. This is stricter failure handling than automatic benchmark retries and must be disclosed if a run cannot complete. Official cloud endorsement/hidden-layout verification and published-baseline checkpoint calibration were not performed.

## What is not included

Model/optimizer/EMA payloads, training data, simulator assets, raw evaluation videos, credentials, authentication caches, browser-control session state, generated UI submissions, and private chat handoffs. Their absence is intentional. Never treat the archive as providing licensed access to gated weights or third-party data.
