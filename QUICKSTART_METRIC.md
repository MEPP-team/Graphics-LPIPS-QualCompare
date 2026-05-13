## Quick Metric Usage

This page is the fastest way to run Graphics-LPIPS-QualCompare on rendered data produced by QualCompare.

If your goal is mainly to reproduce the QualCompare revalidation for *Towards Reproducible Image-based 3D quality assessment: integrated software and new results*, start with [scripts/revalidate_table_qualcompare.bat](scripts/revalidate_table_qualcompare.bat) and the notes in [scripts/README.md](scripts/README.md).

---

## 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

If you use a GPU, install a PyTorch build that matches your CUDA setup before running the command above.

To reproduce the validation workflow, make sure you also have the rendered image dataset locally:

- https://datasets.liris.cnrs.fr/qualcomparerendered-version1

If you prefer to rebuild the rendered inputs, use `QualCompare` with the relevant object datasets for validation.

---

## 2. Prepare the rendered input tree

`Light_GraphicsLPIPS_csv.py` expects a render root with this structure:

```text
<SRC_ROOT>/
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

Important:

- the folder name currently expected by code is `patchs` (not `patches`)
- `<N>VP` must match the `-v` argument (for example `4VP`)

---

## 3. Run a first evaluation

Example command (Windows `cmd` style):

```cmd
python Light_GraphicsLPIPS_csv.py ^
  -m TMQ_NR_4VP_example ^
  -v 4 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db TMQ ^
  -mos ./dataset/TMQ/TMQ_MOS.csv ^
  -testlist ./dataset/TMQ/folds/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6_k0.csv ^
  --src_root D:/RenderedDatasets/TMQ/New_Render/Y_fixed_0.3 ^
  --use_gpu
```

If you use PowerShell, run the same command on one line or adapt line continuation.

---

## 4. Compute correlations

```cmd
python correlation_VP.py ^
  -m TMQ_NR_4VP_example ^
  -v 4 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db TMQ ^
  --out_root ./out
```

---

## 5. Troubleshooting quick checks

- If `Source/<N>VP` or `Distorted/<N>VP` is missing, verify `--src_root`.
- If patch CSV files are not found, verify that `patchs/` exists under each reference object.
- If model loading fails, verify that `./checkpoints/<MODEL_NAME>/latest_net_.pth` exists.
- If GPU errors occur, run without `--use_gpu` or install a matching CUDA/PyTorch build.

---

## Companion workflow note

This repository is designed to consume rendered outputs from QualCompare.

For the bridge and paper-oriented reproduction notes on the QualCompare side, see:

- `QualCompare/docs/graphics_lpips_bridge.md`
- `QualCompare/docs/qomex_reproduction.md`
