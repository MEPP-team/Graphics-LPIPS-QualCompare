# Weighted Graphics-LPIPS Notes

This note summarizes the current experimental additions around fixed baseline
metrics and the proposed patch-weighted Graphics-LPIPS variant.

## Fixed Baseline Revalidation

The fixed baselines are not trained, so they are evaluated once on the full
rendered database, without k-fold splitting.

Scripts:

- `revalidate_lpips.py`: patch-based LPIPS using TorchMetrics.
- `revalidate_ssim.py`: patch-based SSIM using the same QualCompare patches as
  Graphics-LPIPS.
- `revalidate_ssim_images.py`: full-view SSIM, comparing `view_i.png` directly.
- `scripts/revalidate_fixed_baselines_qualcompare.bat`: runs LPIPS, patch-SSIM,
  image-SSIM, then correlations on the paper datasets.

Configured paper datasets in the batch script:

- `TMQ / New_Render / Y_fixed_0.3 / 8VP`
- `TSMD / New_Render / Y_fixed_0.3 / 8VP`
- `SJTU-TMQA / 0_0_light / Y_fixed_0 / 8VP`
- `BASICS / SP / Y_fixed_0 / 4VP`

The batch script now skips a metric if complete result CSVs already exist for
all source objects. This avoids recomputing long runs.

## SSIM Orientation

SSIM is a similarity score:

- `1` means similar / good quality.
- `0` means less similar / worse quality.

Graphics-LPIPS and LPIPS are distances:

- `0` means similar / good quality.
- higher means more different / worse quality.

`correlation_VP.py` keeps SSIM result CSVs unchanged, but converts SSIM to a
distance during correlation:

```text
distance = 1 - SSIM
```

This applies to both `SSIM` and `SSIM_IMAGES`.

## MOS CSV Handling

`correlation_VP.py` now detects MOS columns from CSV headers. This avoids the
TSMD issue where the file uses:

```text
Model,stimulus,MOS
```

instead of the simpler two-column layout.

Recognized object-name columns include:

- `stimulus`
- `ObjectName`
- `name`
- `ppc`

Recognized score columns include:

- `MOS`
- `DMOS`

## Patch-Weighted Graphics-LPIPS Idea

The original Graphics-LPIPS aggregation computes a simple average:

```text
view_score = mean(patch_scores)
object_score = mean(view_scores)
```

The new experimental idea is to learn patch weights:

```text
view_score = sum(weight_i * patch_score_i)
```

The model is weakly supervised: the target is the global MOS, not a true
per-patch importance score.

## Patch Score Export

`Light_GraphicsLPIPS_csv.py` now supports:

```text
--save_patch_scores
```

This writes one CSV per distorted object under:

```text
_PATCH_SCORES_TESTSET_/<reference_object>/<distorted_object>_patch_scores.csv
```

Each row corresponds to one patch and includes:

- object names and MOS
- view index
- patch position and size
- Graphics-LPIPS patch score
- image/content features

Because the feature set has been expanded, old patch-score CSVs should be
regenerated before training the weighting model.

Example TMQ export:

```powershell
python Light_GraphicsLPIPS_csv.py `
  -m TMQ_NR_8VP_yf03_kfolds `
  --use_folds `
  -v 8 `
  -vm Y_fixed_0.3 `
  -rm New_Render `
  -db TMQ `
  -mos ./dataset/TMQ/TMQ_MOS.csv `
  -testlist ./dataset/TMQ/folds/TexturedDB_20_TestList_withnbPatchesPerVP_threth0.6.csv `
  --src_root D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3 `
  --use_gpu `
  --save_patch_scores
```

## Current Patch Features

`train_patch_weighting.py` currently uses 66 features, including:

- Graphics-LPIPS patch score
- patch position and view index
- luminance mean/std
- gradient mean/std/max
- Laplacian variance
- RGB mean/std
- Lab mean/std
- ref/dist difference features
- mask object ratio
- mask edge ratio
- relative position inside the object bounding box

Multi-metric local features, such as local SSIM/L1/L2/PSNR, are not included
yet.

## Training and Applying the Weighting Model

Train:

```powershell
python train_patch_weighting.py `
  --patch_scores_root ./out/TMQ/New_Render/Y_fixed_0.3/TMQ_NR_8VP_yf03_kfolds/8VP `
  --views 8 `
  --output ./checkpoints/WEIGHTED_GLPIPS_TMQ/patch_attention.pt `
  --epochs 30 `
  --use_gpu
```

Apply:

```powershell
python revalidate_weighted_graphicslpips.py `
  -m WEIGHTED_GLPIPS_TMQ `
  --source_model TMQ_NR_8VP_yf03_kfolds `
  --attention_model ./checkpoints/WEIGHTED_GLPIPS_TMQ/patch_attention.pt `
  --patch_scores_root ./out/TMQ/New_Render/Y_fixed_0.3/TMQ_NR_8VP_yf03_kfolds/8VP `
  -v 8 `
  -vm Y_fixed_0.3 `
  -rm New_Render `
  -db TMQ `
  --use_gpu
```

Then compute correlations:

```powershell
python correlation_VP.py `
  -m WEIGHTED_GLPIPS_TMQ `
  -v 8 `
  -vm Y_fixed_0.3 `
  -rm New_Render `
  -db TMQ `
  --out_root ./out
```

## Recommended Scientific Protocol

For a clean experiment, do not train and test the weighting model on the same
examples.

Recommended k-fold protocol:

1. Use the existing Graphics-LPIPS checkpoint for fold `k`.
2. Export patch scores for the train split of fold `k`.
3. Export patch scores for the test split of fold `k`.
4. Train the patch-weighting model only on train patch scores.
5. Apply it to test patch scores.
6. Compute correlations on the test split.
7. Repeat for all folds and average results.

This keeps the comparison with the original Graphics-LPIPS protocol meaningful.
