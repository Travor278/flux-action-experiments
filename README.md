# FLUX Action experiment archive

Source archive of FLUX Action experiments on **RoboTwin** and **RoboDojo**. The owner ended training on 2026-10-22. This is a research/reproduction archive, not a maintained training service or an official benchmark submission.

## What is here

| Directory | Contents |
|---|---|
| `robodojo/scripts/` | Official-history training wrappers, dataset preparation and checks, checkpoint audits, evaluation, and explicitly historical diagnostics |
| `robodojo/policy/` | RGB + measured joint state + instruction adapter |
| `robotwin/scripts/`, `robotwin/configs/` | RoboTwin training, sampling repairs, CPU EMA, export and evaluation workflows |
| `robotwin/historical-local-tools/` | Earlier local helper sources retained for completeness |
| `vendor/flux-action-official/` | Upstream FLUX source at `830f1ce178d7aea52f18fde31d8e8e36ce6bae21` |
| `vendor/robotwin-original/`, `vendor/robotwin-tail/` | Archived experiment variants; **not** the unmodified official-history baseline |
| `docs/` | Provenance, experiment status, licensing and reproduction limits |

**No model weights, optimizer payloads, training datasets, raw videos, credentials, browser sessions, or private conversation handoffs are included.** Obtain upstream weights/data under their own access and license terms. Paths and platform identifiers have been replaced with placeholders.

## Final intended RoboDojo baseline

- Separate single-task models: `cover_blocks` and `plug_in_charger`; 100 official demonstrations per task.
- Pinned official FLUX history implementation: 8 observation ticks, 2 visual snapshots, previous commands, native 25 Hz / 14-dimensional absolute joint actions / 3 RGB cameras.
- Action chunk 32, execute 16, 4 denoising steps. No custom persistent prefix memory or custom temporal-mask loss in this baseline.
- 10,000 optimizer updates, seed 42, 2 H100 per task, microbatch 2 × accumulation 8 × 2 ranks = global batch 32. FP32 master/reduction and BF16 compute, CPU EMA operational wrappers.
- Backbone/head LR `1e-5` / `5e-5`; backbone freeze 200; warmups 600/200; cooldown 9000+1000. These are **experiment choices**, not an official RoboDojo FLUX training recipe.
- Planned evaluation: one fixed final EMA checkpoint per task, official evaluation seeds 0/1/2 × 50 episodes = 150/task, success rate **and** Score, per-seed and mean/std reporting.

See [experiment status](docs/EXPERIMENTS.md). In particular, the final RoboDojo checkpoints and 300-episode evaluation were **not verified complete** before the experiment was discontinued. Do not present the scripts' existence or CPU checks as policy success results.

## Start reading

1. [Reproduction guide and limitations](docs/REPRODUCIBILITY.md)
2. [`eval150_contract.py`](robodojo/scripts/eval150_contract.py): result invariants and pinned official environment configuration assembly.
3. [`prepare_official_r2.py`](robodojo/scripts/prepare_official_r2.py): selected demonstration mapping and native-loader checks.
4. [`launch_train_official_r2.sh`](robodojo/scripts/launch_train_official_r2.sh): initial complete save → exact resume → early save → training/export. Historical receipt paths must be regenerated for a new deployment.
5. [`launch_r2_eval150_pair.sh`](robodojo/scripts/launch_r2_eval150_pair.sh): final-only, six-run evaluation orchestration.

Run the dependency-free contract tests:

```bash
python robodojo/scripts/test_eval150_contract.py
python tools/verify_archive.py
```

These verify archive integrity and evaluation bookkeeping, **not GPU training or simulator success**. Historical experiments (including rejected prefix-memory and cache candidates) remain in the source archive for transparency; they are not the recommended baseline.

## Upstream attribution

- [Black Forest Labs FLUX Action](https://github.com/black-forest-labs/flux-action)
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
- [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

See [third-party notices](docs/THIRD_PARTY.md). Code licenses do not grant rights to model weights, datasets or simulation assets.
