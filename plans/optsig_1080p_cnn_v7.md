# 1080p BSVD optsig v7 — Small CNN predictor plan

**Status: draft for subagent review (2026-05-14).**

## Goal

Build a neural predictor of optimal BSVD σ for 1080p Canon 60D NV-target denoising that **beats constant σ=0.08 by ≥0.3 SSIMU2 mean** on n=19 NV-paired test set with bootstrap-LOCO 5th-percentile > σ=0.08 95th-percentile (proper non-overlapping confidence intervals).

Operator-matching framing: NV is per-clip-tuned in Premiere; "oracle σ" is what reproduces the user's operator settings, NOT blind denoising of noise physics.

## Why we're here (failure history)

Three linear-predictor attempts have all underperformed constant σ=0.08 on NV-target validation:

1. **v3 (production, `sigma_raw + std_intensity + mean_motion`):** beats σ=0.08 by +0.66 SSIMU2 on its own training set (n=33 from this same NV pool). Has training-set advantage — not a clean win.
2. **v4 (multi-scale MAD on Sony+V8):** lost to σ=0.08 by **−7.66 SSIMU2** on n=17. Root cause: 8-bit quantization clips MAD features to constant 0.00291.
3. **v6 (mean+std+motion on SetB 60D + V8):** lost to σ=0.08 by **−2.34 SSIMU2** on n=17. Root cause: BM3D training target ≈ NV target − 0.023 σ. Residual structure (R²_in_sample=0.494) has 2-3 bright-clip outliers that linear features can't separate.

Calibration-shift experiment (May 14): adding LOCO-fit scalar bias to v6 recovers most loss (71.85 SSIMU2) but lands within noise of σ=0.08 (71.97). Confirms the linear features carry no clip-specific signal beyond bias.

## Architecture

**Input:** stack of N=8 random spatial crops at 192×192 from 8 random temporal positions across the clip = 64 RGB tiles per inference. Pre-norm: subtract 0.5, divide by 0.5.

**Backbone (≤200k params, deliberately small to limit overfit):**
- Conv 3→16 (3×3, stride 2) + BN + ReLU
- Conv 16→32 (3×3, stride 2) + BN + ReLU
- Conv 32→64 (3×3, stride 2) + BN + ReLU
- Conv 64→64 (3×3, stride 2) + BN + ReLU
- AdaptiveAvgPool(1) → flatten to 64-d per tile
- Mean over 64 tiles → 64-d clip embedding
- FC 64→32 + Dropout(0.3) + ReLU → 32→7 (logits over the 7-σ grid)

**Output head:** ordinal classification with cross-entropy loss on the 7-σ grid {0.015, 0.025, 0.040, 0.060, 0.080, 0.100, 0.120}. Inference: argmax → σ. **Why classification not regression:** σ-grid is discrete in production anyway (snap-to-grid in v6), residual signal is small enough that classification's bias toward the empirical mode is a feature, not a bug.

**Total params:** ~120k. Order of magnitude below n=clip-frames-per-batch.

## Training data

**Phase A — pretrain on V8 sweep (n=198 rows from 66 SetB 60D clips):**
- Inputs: random crops from the V8-noisy 60-frame slices
- Targets: oracle σ from the L1+SSIM BSVD scoring (per-instance, snap-to-grid)
- Goal: learn the "what σ removes V8-synth noise on BM3D-clean" mapping
- Train/val split: 13/66 clips held out for early-stopping monitor; remaining 53 clips × 3 instances = 159 rows train

**Phase B — fine-tune on n=19 NV-paired clips:**
- Inputs: random crops from the noisy MOV (NOT BM3D-cleaned, NOT V8 — actual real noise the production predictor will see)
- Targets: oracle σ from SSIMU2 scoring of BSVD(noisy, σ) vs NV
- Re-compute oracle σ on the same 7-σ grid (already done in validate_1080p.py output — extract per-clip oracle from validation.log)
- LOCO CV with bootstrap: train on 18 clips, predict held-out clip, repeat × 17 (the 17 that scored; 2 had load failures)
- Lower learning rate (1e-4 vs 5e-4 in pretrain) + frozen first 2 conv blocks
- Hard caps: 50 epochs, early stop on per-fold val loss plateau

## Acceptance gate

Bootstrap-LOCO 1000 resamples:
- v7 mean SSIMU2 5th percentile > σ=0.08 mean SSIMU2 95th percentile (true separation, not just point-mean win)
- AND v7 mean point-estimate > σ=0.08 mean point-estimate by ≥0.3 SSIMU2

If neither met → don't ship. Keep v3 (which already meets the weaker "+0.66 SSIMU2 over σ=0.08 on its own training set" bar).

## Hardware + runtime estimates

- Pretrain (n=198 rows × 8 tiles × 50 epochs): on encoder-host 2070 SUPER ~10 min, on gpu1 4090 ~5 min
- Fine-tune × 17 LOCO folds × 50 epochs (n=18 train) × 64 tiles/clip: ~15-20 min total
- Validation re-run (re-extract oracle σ from existing validation.log — already have it): instant
- Visual A/B at v7 σ vs σ=0.08 σ on MVI_8656, MVI_6174, MVI_8742: 5 min

Estimated end-to-end: ~30 min (excluding subagent review).

## Risks I've identified (subagent: please add more)

1. **n=19 NV-paired is tiny.** Even 120k params can memorize via the spatial-crop augmentation. Mitigations: strong dropout, AdaptiveAvgPool (no spatial bias), aggressive crop/temporal augmentation, classification not regression.

2. **NV is operator-tuned and may not be a deterministic function of the input.** If the same user denoising the same clip on a different day would land on a different σ, no input-conditional model can capture that variance. Spread of optimal σ across visually-similar clips (MVI_0265 = 0.08, MVI_1352 = 0.10, similar mean/std) hints at this.

3. **Phase A pretrain leaks BM3D-target inductive bias.** The CNN learns "this signature → BM3D-optimal σ", which we know is shifted by +0.023 from NV-optimal σ. Phase B has to overpower that prior. Alternatively: include +0.023 σ-shift in Phase A targets so we never learn the BM3D σ in the first place.

4. **The 7-σ grid endpoints (0.015 and 0.120) had ZERO oracle-row mass in the V8 training set.** Classification head's softmax will have flat priors there → predictions never land at endpoints. May be fine (NV oracle distribution also avoids those endpoints) but worth checking.

5. **In-sample R²=0.494 on linear residual fit ≈ in-sample R²=0.65-0.7 for a good linear model. Honest LOCO R² ~0.2-0.3.** That's the ceiling on what input-conditional residual signal exists. A CNN can't manufacture signal that isn't in the data — it can only express more flexible functional forms of existing signal.

6. **Bootstrap-LOCO on n=17 has wide CIs by construction.** Achieving non-overlapping CIs vs σ=0.08 requires the point-estimate to move ~1.5 SSIMU2 over σ=0.08 — much more than the linear-residual analysis suggests is available.

## Acceptable kill criteria during execution

If at any of these checkpoints the result is bad enough, stop and report negative result:
- Phase A val loss > 1.5× train loss after 30 epochs → CNN is memorizing the 53 training clips, not generalizing
- Phase B LOCO mean SSIMU2 < 71.0 after 25 epochs of fine-tune → fundamentally not learning NV signal
- v7 LOCO mean SSIMU2 within 0.3 of v6_calibrated (71.85) → CNN adds nothing over scalar shift

## Files

**To create:**
- `/home/user/bsvd_rocm/optsig1080p_v7/train_v7.py` — pretrain + fine-tune
- `/home/user/bsvd_rocm/optsig1080p_v7/dataset.py` — random spatial+temporal crop loader (Phase A from FFV1 slices, Phase B from MOVs)
- `/home/user/bsvd_rocm/optsig1080p_v7/predictor_v7.pt` — trained weights
- `/home/user/bsvd_rocm/optsig1080p_v7/predictor_v7.onnx` — exported for ORT-MIGraphX deploy
- `/home/user/bsvd_rocm/optsig1080p_v7/eval_v7.py` — LOCO + bootstrap

**To reuse:**
- `/home/user/bsvd_rocm/optsig1080p_v6_run/progress.jsonl` — Phase A training labels
- `/home/user/bsvd_rocm/clips_1080p/test/MVI_*_src.mkv` + `MVI_*_nv.mkv` — Phase B clips
- `/home/user/bsvd_rocm/optsig1080p_v6_run/validation.log` — Phase B oracle σ (parse out per-clip best-SSIMU2 σ)

**To NOT touch:**
- `~/gitproj/bsvd/optimal_sigma_predictor.py` — v3 production stays as default; v7 ships only behind a flag IF accepted
