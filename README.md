
# Graphics-LPIPS-QualCompare

Graphics-LPIPS-QualCompare is a Graphics-LPIPS fork for perceptual quality assessment of textured 3D meshes using rendered multi-view data prepared with QualCompare.

Instead of predicting similarity between isolated 2D image patches only, this repository focuses on:

- patch-based comparison of rendered views of 3D objects
- MOS / DSIS-style supervision
- training on rendered textured-mesh data
- evaluation of distorted objects from one or multiple viewpoints
- correlation analysis between predicted scores and subjective quality scores

This repository is based on the Graphics-LPIPS research line introduced in the paper:
[Textured Mesh Quality Assessment: Large-Scale Dataset and Deep Learning-based Quality Metric](https://yananehme.github.io/publications/2022-ACM-TOG)

This workflow is also associated with the QualCompare revalidation paper:
*Towards Reproducible Image-based 3D quality assessment: integrated software and new results*.

## Quick Links

- **Paper revalidation guide**: [paper_revalidation/README.md](paper_revalidation/README.md) - Full revalidation workflow and results interpretation
- Quick metric usage: [QUICKSTART_METRIC.md](QUICKSTART_METRIC.md) - Fast setup for custom datasets
- Command templates: [scripts/README.md](scripts/README.md)
- Revalidation helper: [paper_revalidation/revalidate_table_qualcompare.bat](paper_revalidation/revalidate_table_qualcompare.bat)

## Repository Focus

The current public codebase is organized around three main entrypoints:

- `train.py`: train a Graphics-LPIPS model on the textured mesh dataset
- `Light_GraphicsLPIPS_csv.py`: evaluate a trained checkpoint on rendered multi-view data using in-memory patch extraction
- `correlation_VP.py`: compute correlation summaries and plots from evaluation outputs

`Light_GraphicsLPIPS_csv.py` is the recommended evaluation script for the current workflow.

## Installation

Install PyTorch and torchvision first, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/MEPP-team/Graphics-LPIPS-QualCompare.git
cd Graphics-LPIPS-QualCompare
```

## Dataset

This metric is only meaningful when used with the associated textured mesh quality dataset. The repository does not ship the dataset itself.

You need two complementary inputs depending on what you want to do:

- the object datasets used as the source for validation and rendering
- the rendered image dataset used directly by the metric workflow

The recommended workflow is now:

1. Download the main textured mesh quality dataset:
   https://datasets.liris.cnrs.fr/textured-mesh-quality-assessment-dataset-version1
2. Download the rendered image dataset used for revalidation and evaluation:
  https://datasets.liris.cnrs.fr/qualcomparerendered-version1
3. If you prefer to regenerate the rendered images yourself, use the companion tool `QualCompare` together with the relevant object datasets for validation.
4. Train or evaluate Graphics-LPIPS-QualCompare on the rendered outputs.

The historical patchified archive from the original project is no longer the primary path for this fork. It is mostly useful for compatibility with the older workflow:

- Historical patchified archive:
  https://perso.liris.cnrs.fr/ynehme/datasets/Graphics-Lpips/dataset.zip

This fork is designed to work alongside the companion rendering pipeline rather than depend on a pre-patchified dataset archive.

## Companion Tool

To fully use the updated workflow, this repository should be used together with `QualCompare`, which prepares the rendered views and patch metadata expected by the current evaluation and training scripts.

Companion repository:

- `QualCompare`: https://github.com/MEPP-team/QualCompare

In practice:

- the main dataset provides the textured 3D objects and associated subjective data
- `QualCompare` renders the objects and produces the image / patch structure expected by this repository
- `Graphics-LPIPS-QualCompare` trains on or evaluates those rendered outputs

### Expected Rendered Structure

The current workflow expects a rendered directory layout of the following form:

```text
<EXPERIMENT_ROOT>/
  Source/
    <N>VP/
      <REFERENCE_OBJECT>/
        views/
          view_1.png
          view_2.png
          ...
        patchs/
          <REFERENCE_OBJECT>_patchlist.csv
  Distorted/
    <N>VP/
      <DISTORTED_OBJECT>/
        views/
          view_1.png
          view_2.png
          ...
```

Important note:

- the folder name is currently `patchs` in the codebase, not `patches`
- `Light_GraphicsLPIPS_csv.py` expects `--src_root` to point to `<EXPERIMENT_ROOT>`
- `train.py` expects `--src_root`, `--root_refPatches`, and `--root_distPatches` to match that rendered structure

## Prepare the Dataset Layout

The scripts expect each dataset to be reachable as `<SRC_ROOT>/Source/<N>VP/` and
`<SRC_ROOT>/Distorted/<N>VP/` (see [Expected Rendered Structure](#expected-rendered-structure)).

However, the published rendered dataset
([qualcomparerendered](https://datasets.liris.cnrs.fr/qualcomparerendered-version1))
extracts to a **flat, per-dataset** layout, without the `Source/<N>VP` wrapper:

```text
<DATASET_ROOT>/
  TMQ_Circle_0.3_8VP_source/<REFERENCE_OBJECT>/{views, masks, patchs}
  TMQ_Circle_0.3_8VP_distorted/<DISTORTED_OBJECT>/{views, masks}
  ...
  dataset_info.json
```

So, **before running training or evaluation**, expose each dataset in the expected
layout. The helper script does this with directory junctions (Windows) or symlinks
(Linux/macOS), so the (tens of GB of) images are **not copied**:

Windows (PowerShell):

```powershell
scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered"
```

Linux/macOS:

```bash
scripts/prepare_dataset_layout.sh /path/to/qualcomparerendered
```

The script reads `dataset_info.json` (or falls back to the 5 known datasets) and
creates `<DATASET_ROOT>/_run/<DB>/Source/<N>VP` and `.../Distorted/<N>VP` for each
dataset, then prints the `--src_root` to use, e.g.:

```text
[ok] TMQ        --src_root "D:\path\to\qualcomparerendered\_run\TMQ"
[ok] TSMD       --src_root "D:\path\to\qualcomparerendered\_run\TSMD"
```

Pass that path to the scripts, for example (TSMD, zero-shot with the shipped
checkpoint):

```bash
python Light_GraphicsLPIPS_csv.py -m TMQ_NR_8VP_yf03_kfolds --use_folds -v 8 \
  -vm Y_fixed_0.3 -rm New_Render -db TSMD \
  -mos ./dataset/TSMD/_TSMD_fulldataset.csv \
  -testlist ./dataset/TSMD/_TSMD_fulldataset.csv \
  --src_root "D:\path\to\qualcomparerendered\_run\TSMD" --use_gpu
```

### For the `paper_revalidation` batch scripts

The `.bat` presets build `SRC_ROOT` as
`<QUALCOMPARE_OUT_ROOT>/<DB>/<RENDER_METHOD>/<VIEW_METHOD>`. Create that deeper
layout with `-ForBat` (Windows) / `--forbat` (Linux), then export the root:

```powershell
scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered" -ForBat
set QUALCOMPARE_OUT_ROOT=D:\path\to\qualcomparerendered\_run
```

```bash
scripts/prepare_dataset_layout.sh /path/to/qualcomparerendered --forbat
export QUALCOMPARE_OUT_ROOT=/path/to/qualcomparerendered/_run
```

See [paper_revalidation/README.md](paper_revalidation/README.md) for the batch workflow.

To undo the junctions later (the source images are never touched):

```powershell
scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered" -Remove
```

## Expected Local Resources

In practice, this repository assumes that:

- `dataset/` contains CSV splits, folds, and dataset metadata used by training and evaluation
- rendered reference and distorted views are available in the directory structure generated by `QualCompare`
- checkpoints are stored in `./checkpoints`
- generated experiment outputs are written to `./out`

Because of file size, licensing, and reproducibility concerns, the dataset is intentionally not bundled in this repository. Download it separately before running training or evaluation.

After setup, a typical working tree looks like:

```text
Graphics-LPIPS-QualCompare/
  data/
  lpips/
  util/
  dataset/
    <dataset csv files and folds>
  checkpoints/
    <trained models>
  out/
    <generated evaluation outputs>
```

At minimum:

- `dataset/` should contain the CSV files referenced by `train.py` and `Light_GraphicsLPIPS_csv.py`
- `checkpoints/` should contain trained model weights when running evaluation
- `out/` will be created by evaluation and correlation scripts

If a public demo checkpoint is provided, it becomes possible to test the full evaluation pipeline without retraining a model first.

## Typical Workflow

### 0. Revalidate or reproduce an experiment

If your goal is mainly to rerun a published or previous experience, start here:

1. Make sure the rendered image dataset is available locally, or regenerate it with `QualCompare` from the object datasets you want to validate.
2. Check [paper_revalidation/README.md](paper_revalidation/README.md) for the paper revalidation pipeline.
3. Run [paper_revalidation/revalidate_table_qualcompare.bat](paper_revalidation/revalidate_table_qualcompare.bat) with `--dry-run` first, then remove `--dry-run` once the configuration matches your setup.

### 1. Prepare the rendered dataset

Download the main textured mesh dataset, then use `QualCompare` to render the reference and distorted objects and generate the patch metadata consumed by this repository.
Don't forget to use the dataset structure described above.

### 2. Train a model

Training is driven by `train.py`.

Example shape:

```bash
python train.py \
  --datasets ./dataset/<TRAIN_SPLIT>.csv \
  --testcsv ./dataset/<TEST_SPLIT>.csv \
  --src_root <SRC_ROOT> \
  --root_refPatches Source/4VP \
  --root_distPatches Distorted/4VP \
  --name <EXPERIMENT_NAME> \
  --target mos \
  --net alex \
  --npatches 150 \
  --nInputImg 4 \
  --nepoch 5 \
  --nepoch_decay 5 \
  --use_gpu
```

Training writes checkpoints under `./checkpoints/<EXPERIMENT_NAME>/`.

Concrete example:

```bash
python train.py \
  --datasets ./dataset/TMQ/folds/TMQ_train_k0.csv \
  --testcsv ./dataset/TMQ/folds/TMQ_test_k0.csv \
  --src_root D:/RenderedDatasets/TMQ/New_Render/Y_fixed_0.3 \
  --root_refPatches Source/4VP \
  --root_distPatches Distorted/4VP \
  --name TMQ_NR_4VP_example \
  --target mos \
  --net alex \
  --npatches 150 \
  --nInputImg 4 \
  --nepoch 5 \
  --nepoch_decay 5 \
  --use_gpu
```

### 3. Evaluate a checkpoint

The main evaluation path is `Light_GraphicsLPIPS_csv.py`.

Example shape:

```bash
python Light_GraphicsLPIPS_csv.py \
  -m <MODEL_NAME> \
  -v <N_VIEWS> \
  -vm <VIEW_METHOD> \
  -rm <RENDER_METHOD> \
  -db <DATABASE_NAME> \
  -mos ./dataset/<MOS_FILE>.csv \
  -testlist ./dataset/<TEST_LIST>.csv \
  --src_root <EXPERIMENT_ROOT> \
  --use_gpu
```

This script reconstructs patches in memory from rendered views instead of requiring a fully materialized patch dataset on disk for every evaluation run.

`--src_root` should point to the directory that contains `Source/<N>VP/` and `Distorted/<N>VP/`.

Concrete example:

```bash
python Light_GraphicsLPIPS_csv.py \
  -m TMQ_NR_4VP_example \
  -v 4 \
  -vm Y_fixed_0.3 \
  -rm New_Render \
  -db TMQ \
  -mos ./dataset/TMQ/TMQ_MOS.csv \
  -testlist ./dataset/TMQ/folds/TMQ_test_k0.csv \
  --src_root D:/RenderedDatasets/TMQ/New_Render/Y_fixed_0.3 \
  --use_gpu
```

### 4. Compute correlations

After evaluation, run:

```bash
python correlation_VP.py \
  -m <MODEL_NAME> \
  -v <N_VIEWS> \
  -vm <VIEW_METHOD> \
  -rm <RENDER_METHOD> \
  -db <DATABASE_NAME> \
  --out_root ./out
```

Outputs are typically written under `./out/...`.

Concrete example:

```bash
python correlation_VP.py \
  -m TMQ_NR_4VP_example \
  -v 4 \
  -vm Y_fixed_0.3 \
  -rm New_Render \
  -db TMQ \
  --out_root ./out
```

## Ready-to-Adapt Command Files

The `scripts/` directory contains command templates that can be copied and adapted to a local setup:

- `scripts/train_metric.txt`
- `scripts/evaluate_metric.txt`
- `scripts/correlate_metric.txt`

See also:

- `scripts/README.md` for command-template notes
- `paper_revalidation/README.md` for the complete paper revalidation pipeline

## Quick Patch-Level Example

For a simple sanity check between two patches:

```bash
python GraphicsLpips_2imgs.py \
  -p0 imgs/ex_ref.png \
  -p1 imgs/ex_p0.png \
  --use_gpu
```

## Repository Layout

- `train.py`: training entrypoint
- `Light_GraphicsLPIPS_csv.py`: current evaluation entrypoint
- `correlation_VP.py`: post-processing and correlation analysis
- `data/`: data loading pipeline
- `lpips/`: local LPIPS implementation and trainer
- `util/`: visualization and utility helpers
- `scripts/`: reusable command templates for training, evaluation, and correlation
- `paper_revalidation/`: paper-oriented revalidation pipelines and fixed baseline scripts

## Notes

- Several scripts still reflect a research workflow and may contain environment-specific assumptions.
- The current workflow assumes that rendered views and patch metadata come from the companion `QualCompare` pipeline.
- `dataset/`, `checkpoints/`, and `out/` are runtime resources and are not committed here.

## QualCompare Revalidation

This repository includes the pre-trained checkpoint `TMQ_NR_8VP_yf03_kfolds`, which enables direct evaluation of Graphics-LPIPS-QualCompare on new datasets without retraining. This is the checkpoint used for the QualCompare revalidation workflow associated with *Towards Reproducible Image-based 3D quality assessment: integrated software and new results*.

### Quick Start for Revalidation

The easiest way to reproduce paper results is using the helper script:

```bash
# First, set the root containing your QualCompare renders
set QUALCOMPARE_OUT_ROOT=D:\path\to\QualCompare\out
paper_revalidation\revalidate_table_qualcompare.bat --dry-run

# Review the commands, then run without --dry-run
paper_revalidation\revalidate_table_qualcompare.bat
```

### Zero-Shot Evaluation Workflow

To evaluate the pre-trained `TMQ_NR_8VP_yf03_kfolds` checkpoint on a custom dataset:

1. **Prepare rendered views** with QualCompare from your 3D objects
2. **Structure rendered data** according to the expected layout (see below)
3. **Run evaluation** with `Light_GraphicsLPIPS_csv.py`
4. **Compute correlations** with `correlation_VP.py`

### Expected Image Directory Hierarchy

The metric expects rendered images organized in a specific structure:

```
<SRC_ROOT>/
├── Source/
│   ├── 8VP/                          # Number of views must match -v argument
│   │   ├── reference_obj_1/
│   │   │   ├── views/
│   │   │   │   ├── view_1.png
│   │   │   │   ├── view_2.png
│   │   │   │   └── ...
│   │   │   └── patchs/
│   │   │       └── reference_obj_1_patchlist.csv
│   │   ├── reference_obj_2/
│   │   │   └── ...
│   │   └── ...
│   └── <OTHER_VP>/                  # e.g., 4VP for 4-view models
│       └── ...
│
└── Distorted/
    ├── 8VP/
    │   ├── distorted_obj_1_v1/
    │   │   └── views/
    │   │       ├── view_1.png
    │   │       ├── view_2.png
    │   │       └── ...
    │   ├── distorted_obj_1_v2/
    │   │   └── ...
    │   └── ...
    └── <OTHER_VP>/
        └── ...
```

**Important notes:**

- Use `patchs/` (not `patches/`): this is the folder name expected by the code
- Patch CSV files are only required in Source, not in Distorted
- The number of views (e.g., `8VP`) must match the `-v` argument in evaluation commands
- Images must be PNG format
- All objects under the same `<N>VP` folder must have the same number of views

### Example: Evaluate on Custom Data

```bash
# Set up your rendered structure at, e.g., D:\MyRenders
# Then run:

python Light_GraphicsLPIPS_csv.py ^
  -m TMQ_NR_8VP_yf03_kfolds ^
  -v 8 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db CustomDB ^
  -mos ./dataset/CustomDB/mos_scores.csv ^
  -testlist ./dataset/CustomDB/test_list.csv ^
  --src_root D:\MyRenders ^
  --use_gpu
```

Then compute correlations:

```bash
python correlation_VP.py ^
  -m TMQ_NR_8VP_yf03_kfolds ^
  -v 8 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db CustomDB ^
  --out_root ./out
```

### Dataset CSV Format

- **MOS CSV** (`-mos`): Required format with columns `[object_name, mos_score]`
- **Test List** (`-testlist`): Required format with columns `[object_name]` or similar, listing objects to evaluate
- **Training CSV** (`train.py --datasets` / `--testcsv`): Required format with columns `Model,stimulus,MOS`, where `Model` is the reference object and `stimulus` is the distorted object folder name.

Some dataset CSVs store MOS values that have been normalized for GraphicsLPIPS training. In that training setup, the target follows a distortion-distance convention: `0` means close to the reference and `1` means strongly distorted. When a source dataset provides quality MOS values where higher means better quality, those scores should be rescaled and inverted before training, for example:

```text
MOS_training = 1 - (MOS_original / 100)
```

### Checkpoint Details

`TMQ_NR_8VP_yf03_kfolds`:
- Trained on the Textured Mesh Quality (TMQ) dataset
- Uses 8 views per object (`8VP`)
- Fixed Y camera angle with 0.3 units height offset
- Contains multiple fold checkpoints for k-fold validation
- Weights can be found in `./checkpoints/TMQ_NR_8VP_yf03_kfolds/fold_k*/latest_net_.pth`

## Reference

Gautier Campagne, Florent Dupont, Guillaume Lavoué, Florence Denis, Johanna Delanoy, "Towards Reproducible Image-based 3D quality assessment: integrated software and new results".

Yana Nehme, Johanna Delanoy, Florent Dupont, Jean-Philippe Farrugia, Patrick Le Callet, Guillaume Lavoue, "Textured Mesh Quality Assessment: Large-Scale Dataset and Deep Learning-based Quality Metric".

## License

This project is distributed under the Mozilla Public License v. 2.0. See `LICENSE-MPL2.txt`.




