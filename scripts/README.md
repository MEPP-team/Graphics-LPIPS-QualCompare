# Command Templates

The files in this directory are ready-to-adapt command templates for the main Graphics-LPIPS workflow.

Files:

- `train_metric.txt`: train a model
- `evaluate_metric.txt`: evaluate a checkpoint
- `correlate_metric.txt`: compute correlation summaries
- `revalidate_table_qualcompare.bat`: end-to-end table revalidation helper (QualCompare renders -> metric -> correlation)

For paper-oriented revalidation of the new Graphics-LPIPS rows, start with:

```cmd
scripts\revalidate_table_qualcompare.bat --dry-run
```

Then edit the configuration section in the `.bat` file (notably `SRC_ROOT` and preset values), and run it without `--dry-run`.

## Revalidation Script Specifications

The helper script `revalidate_table_qualcompare.bat` now runs a full pipeline:

1. Training (`train.py`) to generate checkpoints
2. Evaluation (`Light_GraphicsLPIPS_csv.py`) on rendered QualCompare views
3. Correlation summary (`correlation_VP.py`)

Main configuration variables in the script:

- `SRC_ROOT`: QualCompare render root (must contain `Source/<N>VP` and `Distorted/<N>VP`)
- `RUN_TRAINING`: `1` to train before evaluation, `0` to reuse existing checkpoints
- `KEEP_ONLY_LATEST`: `1` to remove epoch snapshots and keep only `latest_net_.pth`
- `USE_FOLDS`: `1` for k-fold mode, `0` for single split mode
- `USE_GPU`: enables `--use_gpu` in evaluation

TMQ fold CSV naming used by the current script:

- train base: `dataset/TMQ/folds/TexturedDB_80_TrainList_withnbPatchesPerVP_threth0.6.csv`
- test base: `dataset/TMQ/folds/TexturedDB_20_TestList_withnbPatchesPerVP_threth0.6.csv`
- fold files expected by scripts: `_k0` to `_k4` suffixes

Checkpoint policy:

- evaluation scripts consume `latest_net_.pth`
- keeping only latest is valid for reproducibility runs
- keep epoch snapshots only if you need rollback/diagnostic comparisons

Typical run sequence:

```cmd
scripts\revalidate_table_qualcompare.bat --dry-run
scripts\revalidate_table_qualcompare.bat
```

These examples use Windows `cmd` line continuation with `^`.

If you are using:

- `cmd.exe`: you can copy them as-is
- PowerShell: place the command on a single line, or replace `^` with PowerShell-friendly line continuation/backtick syntax
- Bash: place the command on a single line, or replace `^` with `\`

Main placeholders to adapt:

- dataset CSV paths under `./dataset/...`
- `--src_root`, which should point to the rendered dataset produced by `QualCompare`
- model name passed with `--name` during training and `-m` during evaluation
- database / render / viewpoint settings such as `TMQ`, `New_Render`, `Y_fixed_0.3`, and `4`

Expected rendered structure:

```text
<EXPERIMENT_ROOT>/
  Source/
    <N>VP/
      <REFERENCE_OBJECT>/
        views/
        patchs/
  Distorted/
    <N>VP/
      <DISTORTED_OBJECT>/
        views/
```

Important note:

- the code currently expects the folder name `patchs`
