# Paper Revalidation Pipeline

This folder contains the scripts used to reproduce or extend the paper
revalidation experiments. They are intentionally separated from the core
Graphics-LPIPS-QualCompare metric code.

Core metric workflow:

- `train.py`
- `Light_GraphicsLPIPS_csv.py`
- `correlation_VP.py`

Paper revalidation workflow:

- `revalidate_table_qualcompare.bat`: trained Graphics-LPIPS rows, from training or existing checkpoints to correlations
- `revalidate_fixed_baselines_qualcompare.bat`: fixed baselines (`LPIPS_TORCHMETRICS`, `SSIM`, `SSIM_IMAGES`)
- `revalidate_lpips.py`, `revalidate_ssim.py`, `revalidate_ssim_images.py`: baseline evaluators
- `run_fixed_baselines_then_shutdown.bat`: optional long-run helper

## Required Inputs

Before running the pipeline, make sure these resources exist locally:

- rendered data produced by QualCompare
- dataset CSV files under `dataset/`
- checkpoints under `checkpoints/` when `RUN_TRAINING=0`
- a Python environment with the dependencies from `requirements.txt`

The rendered data must follow this structure:

```text
<QUALCOMPARE_OUT_ROOT>/
  <DATABASE>/
    <RENDER_METHOD>/
      <VIEW_METHOD>/
        Source/
          <N>VP/
            <REFERENCE_OBJECT>/
              views/
                view_1.png
              patchs/
                <REFERENCE_OBJECT>_patchlist.csv
        Distorted/
          <N>VP/
            <DISTORTED_OBJECT>/
              views/
                view_1.png
```

The folder name is `patchs`, matching the current codebase.

## Recommended Setup

Set `QUALCOMPARE_OUT_ROOT` to the folder that contains the rendered dataset
directories:

```cmd
set QUALCOMPARE_OUT_ROOT=D:\path\to\QualCompare\out
```

PowerShell equivalent:

```powershell
$env:QUALCOMPARE_OUT_ROOT = "D:\path\to\QualCompare\out"
```

Both batch scripts automatically use `.venv\Scripts\python.exe` when it exists
at the repository root. Otherwise they fall back to `python`.

## Graphics-LPIPS Revalidation

Always start with a dry run:

```cmd
paper_revalidation\revalidate_table_qualcompare.bat --dry-run --preset WPC_SP_CIRCLE_5FOLD
```

Then run the pipeline:

```cmd
paper_revalidation\revalidate_table_qualcompare.bat --preset WPC_SP_CIRCLE_5FOLD
```

This script can:

1. train a Graphics-LPIPS checkpoint when `RUN_TRAINING=1`
2. evaluate a checkpoint with `Light_GraphicsLPIPS_csv.py`
3. compute correlations with `correlation_VP.py`

Available presets:

- `TMQ_5FOLD`
- `TSMD_5FOLD`
- `TSMD_ZEROSHOT`
- `SJTU_TMQA_5FOLD`
- `SJTU_TMQA_ZEROSHOT`
- `BASICS_5FOLD_4VP`
- `BASICS_ZEROSHOT`
- `WPC_5FOLD`
- `WPC_ZEROSHOT`
- `WPC2_5FOLD`
- `WPC2_ZEROSHOT`
- `WPC_SP_CIRCLE_5FOLD`
- `WPC_SP_CIRCLE_TMQ_ZEROSHOT`

## Fixed Baseline Revalidation

Dry run:

```cmd
paper_revalidation\revalidate_fixed_baselines_qualcompare.bat --dry-run --preset WPC_SP_CIRCLE
```

Run:

```cmd
paper_revalidation\revalidate_fixed_baselines_qualcompare.bat --preset WPC_SP_CIRCLE
```

The fixed-baseline script evaluates:

- `LPIPS_TORCHMETRICS`
- `SSIM`
- `SSIM_IMAGES`

These baselines are not trained. They run once on the selected full test set.

## Outputs

Outputs are written under:

```text
out/<DATABASE>/<RENDER_METHOD>/<VIEW_METHOD>/<MODEL>/<N>VP/
```

Common summary files:

- `correlation_summary_kfolds.csv`
- `correlation_folds_stats.csv`

Per-object metric outputs are stored in `_METRIC_RESULTS_TESTSET_`.

## Troubleshooting

Missing `Source/<N>VP` or `Distorted/<N>VP`:

- check `QUALCOMPARE_OUT_ROOT`
- check the selected preset values for `DATABASE`, `RENDER_METHOD`, and `VIEW_METHOD`
- confirm that QualCompare generated the expected render tree

Missing `<REFERENCE_OBJECT>_patchlist.csv`:

- regenerate patches with QualCompare
- confirm the folder is named `patchs`

Missing checkpoint:

- either set `RUN_TRAINING=1`
- or place `latest_net_.pth` under `checkpoints/<MODEL_NAME>/`
- for k-fold checkpoints, place one file per fold under `checkpoints/<MODEL_NAME>/fold_k0/`, ..., `fold_k4/`

Missing `torchmetrics`:

- install the project dependencies in the active environment
- or create `.venv` at the repository root so the batch scripts pick it up automatically

Unexpected distorted folders in the output:

- provide the correct `-testlist` CSV
- `Light_GraphicsLPIPS_csv.py` and the fixed baseline scripts filter distorted objects from that CSV when it is provided

Existing results are skipped by the fixed-baseline script:

- pass `--force` when running an individual baseline script manually
- or delete the corresponding output folder before rerunning the batch pipeline

For Graphics-LPIPS evaluation, rerun with a different model/output name or remove
the previous output folder if you need a clean recomputation.
