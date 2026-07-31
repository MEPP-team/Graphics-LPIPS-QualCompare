# Graphics-LPIPS-QualCompare

Graphics-LPIPS-QualCompare is a fork of [Graphics-LPIPS](https://github.com/MEPP-team/Graphics-LPIPS), a perceptual quality metric for 3D graphics. Graphics-LPIPS-QualCompare aims at facilitating a controlled evaluation (training and testing) on any datasets, especially using multi-view renderings produced by the companion tool
[QualCompare](https://github.com/MEPP-team/QualCompare).


## How the metric works

For each object, patches are sampled on the **reference** views (the patch lists
produced by QualCompare) and read at the same coordinates from the **distorted**
views. The score is aggregated in three steps:

1. per patch: a learned Graphics-LPIPS distance (AlexNet backbone + trained linear
   layers);
2. per view: the mean over its patches;
3. per object: the mean over the rendered views.

Object scores are mapped to the subjective scale with a binomial GLM; PLCC and
SROCC are computed after mapping. Under k-fold cross-validation, the reported PLCC
is the mean of the per-fold values.

## Repository entry points

| Script | Role |
|---|---|
| `Light_GraphicsLPIPS_csv.py` | Evaluate a checkpoint on rendered views (patches reconstructed in memory). Recommended evaluation path. |
| `train.py` | Train a Graphics-LPIPS model on rendered data. |
| `correlation_VP.py` | Compute PLCC/SROCC summaries from evaluation outputs. |

## Requirements

- **Python 3.12** (paper environment: 3.12.10).
- **An NVIDIA GPU with CUDA.** Training and evaluation are CUDA-only in the current
  code (CPU execution is not supported).
- Python dependencies from `requirements.txt`. The paper environment is
  PyTorch 2.7.0 (CUDA 12.8, cuDNN 9.7.1) and OpenCV 4.11.0.86.

## Installation

```bash
git clone https://github.com/MEPP-team/Graphics-LPIPS-QualCompare.git
cd Graphics-LPIPS-QualCompare

# Install the CUDA build of PyTorch first (PyPI ships the CPU build):
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Pretrained checkpoint

The repository ships `checkpoints/TMQ_NR_8VP_yf03_kfolds`, trained on the Textured
Mesh Quality (TMQ) dataset with 8 views per object and the `Y_fixed` camera at a
0.3 height offset. It contains **five fold checkpoints**:

```text
checkpoints/TMQ_NR_8VP_yf03_kfolds/fold_k0/latest_net_.pth
...                               /fold_k4/latest_net_.pth
```

There is no single top-level weight file, so this checkpoint must be used with the
`--use_folds` flag (evaluation runs each fold and averages the results). It enables
zero-shot evaluation on other datasets without retraining.

## Data

The datasets are not bundled (size and licensing). The repository ships only the
CSV splits, folds and MOS files under `dataset/`.

The companion dataset <https://datasets.liris.cnrs.fr/qualcomparerendered-version1>
provides rendered views, masks and patch data for 5 mesh and point cloud datasets
(TMQ, TSMD, SJTU-TMQA, BASICS and WPC) that are suited to be used with Graphics-LPIPS-QualCompare.

Alternatively, renders can be regenerated with [QualCompare](https://github.com/MEPP-team/QualCompare) from the source objects.

### Expected layout

The scripts read each dataset from `<SRC_ROOT>/Source/<N>VP/` and
`<SRC_ROOT>/Distorted/<N>VP/`:

```text
<SRC_ROOT>/
  Source/<N>VP/<REFERENCE_OBJECT>/
    views/   view_1.png, view_2.png, ...
    patchs/  <REFERENCE_OBJECT>_patchlist.csv
  Distorted/<N>VP/<DISTORTED_OBJECT>/
    views/   view_1.png, view_2.png, ...
```

- The folder is named `patchs` (not `patches`).
- Patch CSVs are required only under `Source/`.
- `<N>VP` must match the `-v` argument; images must be PNG.

### Preparing the layout from the published dataset

The published rendered dataset extracts to a **flat** per-dataset layout
(`<NAME>_source/`, `<NAME>_distorted/`) that does **not** match the structure
above. Map it with the helper script, which creates directory junctions (Windows)
or symlinks (Linux/macOS) — no image is copied:

```powershell
scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered"
```

```bash
scripts/prepare_dataset_layout.sh /path/to/qualcomparerendered
```

The script reads `dataset_info.json`, creates
`<DATASET_ROOT>/_run/<DB>/Source/<N>VP` (and `Distorted/<N>VP`) for each dataset,
and prints the `--src_root` to pass to the Python scripts:

```text
[ok] TMQ        --src_root "D:\path\to\qualcomparerendered\_run\TMQ"
[ok] TSMD       --src_root "D:\path\to\qualcomparerendered\_run\TSMD"
```

For the `paper_revalidation` batch scripts, use `-ForBat` / `--forbat` instead (it
builds the deeper `<DB>/<RENDER_METHOD>/<VIEW_METHOD>` layout those scripts expect)
and set `QUALCOMPARE_OUT_ROOT` to the printed `_run` folder. Add `-Remove` to
delete the junctions later.

## Usage

### Quick check (two images)

Verify the metric and the shipped checkpoint on two bundled example images (no
dataset required; runs on CPU):

```bash
python GraphicsLpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png
```

It prints a Graphics-LPIPS distance in `[0, 1]` (~0.53 for the reference vs its
heavily JPEG-compressed version).

The dataset commands below require a CUDA GPU (`--use_gpu`).

### Evaluate a checkpoint

Example — zero-shot evaluation of the shipped TMQ checkpoint on TSMD:

```bash
python Light_GraphicsLPIPS_csv.py -m TMQ_NR_8VP_yf03_kfolds --use_folds -v 8 -vm Y_fixed_0.3 -rm New_Render -db TSMD -mos ./dataset/TSMD/_TSMD_fulldataset.csv -testlist ./dataset/TSMD/_TSMD_fulldataset.csv --src_root "<SRC_ROOT>/TSMD" --use_gpu
```

Per-object metric CSVs are written under
`out/<DB>/<RENDER_METHOD>/<VIEW_METHOD>/<MODEL>/<N>VP/`.

### Compute correlations

```bash
python correlation_VP.py -m TMQ_NR_8VP_yf03_kfolds --use_folds -v 8 -vm Y_fixed_0.3 -rm New_Render -db TSMD --out_root ./out
```

This writes `correlation_folds_stats.csv` (per-fold and mean PLCC/SROCC) in the
experiment directory.

### Train a model

Example — 5-fold training on TMQ (8 views); `--use_folds` appends `_kX` to the
train/test CSV names:

```bash
python train.py --name TMQ_8VP_example --use_folds --src_root "<SRC_ROOT>/TMQ" --datasets ./dataset/TMQ/folds/TexturedDB_80_TrainList_withnbPatchesPerVP_threth0.6.csv --testcsv ./dataset/TMQ/folds/TexturedDB_20_TestList_withnbPatchesPerVP_threth0.6.csv --root_refPatches Source/8VP --root_distPatches Distorted/8VP --target mos --net alex --npatches 150 --nInputImg 4 --nepoch 5 --nepoch_decay 5 --use_gpu
```

Checkpoints are written under `checkpoints/TMQ_8VP_example/fold_k*/`. Folds whose
directory already exists are skipped unless `--overwrite` is given.

Copy-and-adapt single-line templates are provided in `scripts/`
(`train_metric.txt`, `evaluate_metric.txt`, `correlate_metric.txt`); see
[scripts/README.md](scripts/README.md).

## Reproducing the paper

The `paper_revalidation/` folder automates the paper's table (trained and zero-shot
rows) and the fixed baselines (LPIPS via `torchmetrics`, SSIM via `scikit-image`).

```powershell
scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered" -ForBat
set QUALCOMPARE_OUT_ROOT=D:\path\to\qualcomparerendered\_run
paper_revalidation\revalidate_table_qualcompare.bat --dry-run --preset TSMD_ZEROSHOT
paper_revalidation\revalidate_table_qualcompare.bat --preset TSMD_ZEROSHOT
```

See [paper_revalidation/README.md](paper_revalidation/README.md) for the full list
of presets and the fixed-baseline pipeline.

## Dataset CSV formats

- **Training CSV** (`--datasets` / `--testcsv`): columns `Model,stimulus,MOS`, where
  `Model` is the reference object and `stimulus` is the distorted object folder name.
- **MOS CSV** (`-mos`) and **test list** (`-testlist`): the reference/stimulus and
  MOS columns are detected from the header (e.g. `stimulus`/`name`/`ppc` and
  `mos`/`dmos`).

Training targets follow a distortion-distance convention (`0` = close to the
reference, `1` = strongly distorted). Source MOS scales differ across datasets, so
values must be rescaled/inverted accordingly before training; the CSVs shipped
under `dataset/` are already prepared.

## Repository layout

```text
train.py                    training entry point
Light_GraphicsLPIPS_csv.py  evaluation entry point
correlation_VP.py           correlation analysis
ssim.py                     SSIM baseline wrapper
data/                       data-loading pipeline
lpips/                      Graphics-LPIPS network and trainer
util/                       visualization and helpers
scripts/                    command templates and prepare_dataset_layout
paper_revalidation/         paper reproduction pipelines and fixed baselines
dataset/                    CSV splits, folds and MOS files (runtime data not committed)
checkpoints/                model weights (only the shipped TMQ checkpoint is committed)
out/                        evaluation and correlation outputs (created at runtime)
```

## Citation

Paper under review

## License

Distributed under the Mozilla Public License 2.0. See `LICENSE-MPL2.txt`.
