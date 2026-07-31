# Command Templates

This folder contains ready-to-adapt command templates for the main
Graphics-LPIPS-QualCompare metric workflow.

Files:

- `train_metric.txt`: train a model
- `evaluate_metric.txt`: evaluate a checkpoint
- `correlate_metric.txt`: compute correlation summaries

These templates are examples for users who want to run the metric on their own
rendered data. They are not the paper revalidation pipeline.

Related documentation:

- [Quick metric usage](../QUICKSTART_METRIC.md)
- [Paper revalidation pipeline](../paper_revalidation/README.md)

## Placeholders

Adapt these values before running the commands:

- dataset CSV paths under `dataset/`
- `--src_root`, which must point to a QualCompare render root
- model name passed with `--name` during training and `-m` during evaluation
- database, render method, view method, and number of views

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

The folder name is currently `patchs`, not `patches`.

Each `.txt` template is a single-line command (preceded by `#` comment lines
explaining the `<...>` placeholders), so it can be copied and pasted as-is into
`cmd`, PowerShell or Bash — no line-continuation characters to adapt.
