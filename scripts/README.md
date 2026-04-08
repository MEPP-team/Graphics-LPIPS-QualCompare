# Command Templates

The files in this directory are ready-to-adapt command templates for the main Graphics-LPIPS workflow.

Files:

- `train_metric.txt`: train a model
- `evaluate_metric.txt`: evaluate a checkpoint
- `correlate_metric.txt`: compute correlation summaries

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
