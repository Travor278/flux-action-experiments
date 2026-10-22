# Experiment history and evidence limits

## RoboTwin

Earlier `click_bell` / `handover_block` experiments used 50 clean demonstrations per task and 10k updates. The repaired pipeline retained all same-episode source instructions and added explicit masks for synthetic tail padding. This is an experiment-specific modification, not an unmodified upstream FLUX implementation.

Historical internal audits recorded repaired final results of **92/100 click_bell** and **26/100 handover_block**. The handover evaluation resumed after interruption while preserving 33 already-counted episodes. Those figures concern the repaired RoboTwin experiment only; no isolated ablation establishes that one repair alone caused the gains. Raw result/videos are not distributed in this code archive, and these are not externally verified leaderboard claims.

## RoboDojo: rejected/abandoned variants

- A custom persistent visual prefix at ticks 0/25/50 was implemented and functionally tested. The owner subsequently requested the official history approach. This extension was abandoned; its implementation/patch installers remain as historical code.
- Frozen-VAE caching passed a local token comparison but failed the full model/optimizer/EMA fingerprint acceptance check. It was **not adopted**.
- Conveyor preparation exposed a simulation graph compatibility issue. That historical anonymous-stage repair is not part of Cover/Charger evaluation.

## Final intended RoboDojo round

On 2026-10-21 the owner requested fresh `cover_blocks` training plus `plug_in_charger`, excluding Conveyor. Both were submitted with 100 demonstrations and a 10k-update budget each. Early complete checkpoint saving and exact resumption were verified. The last retained local monitoring evidence was 310/290 updates, not final completion.

The owner later authorized strengthening evaluation reliability:

- one fixed final checkpoint; three official layout-seed banks;
- 50 episodes per bank per task (300 total across the two tasks);
- independent success count and official progress Score audit;
- frozen task/runtime/config/layout provenance;
- reject duplicate/missing layouts, mixed checkpoints, malformed scores and incomplete seed aggregation;
- preserve partial runs and require diagnosed resumption; no silent scene replacement.

CPU/source/configuration checks passed. **No verified final RoboDojo evaluation result is claimed.** Three evaluation seeds of one model measure evaluation variability, not variability across three independently trained models.

## Closeout, 2026-10-22

The owner discontinued training and requested public code archival followed by remote cleanup. On inspection, the RoboDojo HDD and SSD experiment directories were already absent. This archive therefore uses retained local authored RoboDojo source, while the remaining RoboTwin/FLUX source was recovered directly from the remote workspace before cleanup. Missing remote-only receipts/config bytes are not fabricated. Reconstructed templates are explicitly labelled as such.
