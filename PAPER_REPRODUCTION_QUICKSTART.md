# Paper Results Reproduction - Quick Start

This guide walks you through reproducing the Graphics-LPIPS paper results using the pre-trained checkpoint.

## Prerequisites

1. **Python 3.8+** with PyTorch installed
2. **Dependencies** from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
3. **Rendered image dataset** prepared with QualCompare
   - If you have raw 3D objects, use QualCompare to render them first
   - See [QualCompare documentation](https://gitlab.liris.cnrs.fr/gcampagne/qualcompare)

## 5-Minute Setup

### Option A: Interactive Notebook (Recommended for First Time)

```bash
# Start Jupyter and open the reproduction notebook
jupyter notebook reproduce_paper_results.ipynb
```

Then follow the notebook steps:
1. Update `SRC_ROOT` path to point to your rendered data
2. Run validation to check your directory structure
3. Run evaluation
4. Compute correlations and results

### Option B: Batch Script (For Reproducible Workflow)

```bash
# Edit the configuration in this script first
scripts\revalidate_table_qualcompare.bat --dry-run

# Review the output, then run without --dry-run
scripts\revalidate_table_qualcompare.bat
```

### Option C: Manual Commands (For Custom Setups)

```bash
# Step 1: Evaluate the checkpoint on your rendered data
python Light_GraphicsLPIPS_csv.py ^
  -m TMQ_NR_8VP_yf03_kfolds ^
  -v 8 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db YourDatabaseName ^
  -mos ./dataset/YourDB/mos_scores.csv ^
  -testlist ./dataset/YourDB/test_list.csv ^
  --src_root D:\path\to\rendered\images ^
  --use_gpu

# Step 2: Compute correlations
python correlation_VP.py ^
  -m TMQ_NR_8VP_yf03_kfolds ^
  -v 8 ^
  -vm Y_fixed_0.3 ^
  -rm New_Render ^
  -db YourDatabaseName ^
  --out_root ./out
```

## Expected Rendered Directory Structure

Your rendered images must be organized like this:

```
<YOUR_RENDERS>/
├── Source/
│   └── 8VP/
│       ├── reference_obj_1/
│       │   ├── views/
│       │   │   ├── view_1.png
│       │   │   ├── view_2.png
│       │   │   └── ... (8 views total)
│       │   └── patchs/
│       │       └── reference_obj_1_patchlist.csv
│       ├── reference_obj_2/
│       └── ...
│
└── Distorted/
    └── 8VP/
        ├── distorted_obj_1_variant1/
        │   └── views/
        │       └── view_*.png
        └── ...
```

**Important:** Use folder name `patchs` (not `patches`)

## What You'll Get

After running the pipeline, you'll have:

1. **Per-object predictions**: CSV files with predicted quality scores
2. **Correlation summary**: PLCC (Pearson) and SROCC (Spearman) values
3. **Scatter plots**: Visualization of predicted vs subjective scores
4. **Statistical summaries**: In `./out/` directory

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
pip install torch torchvision  # Ensure PyTorch is installed
```

### "Checkpoint not found" error
- Verify `./checkpoints/TMQ_NR_8VP_yf03_kfolds/` exists
- Check `fold_k*/latest_net_.pth` weights files are present

### "No such file or directory: views/ or patchs/"
- Check your directory structure matches the expected layout
- Run the validation helper in the notebook for detailed diagnostics

### GPU memory errors
- Remove `--use_gpu` flag to run on CPU
- Or reduce batch size in `Light_GraphicsLPIPS_csv.py` configuration

## Full Documentation

- **README.md**: Complete architecture and workflow documentation
- **QUICKSTART_METRIC.md**: Detailed metric evaluation guide
- **scripts/README.md**: Script configuration and usage notes
- **reproduce_paper_results.ipynb**: Interactive step-by-step workflow

## Paper Reference

Nehme, Y., Delanoy, J., Dupont, F., Farrugia, J., Le Callet, P., Lavoue, G. (2022).
"Textured Mesh Quality Assessment: Large-Scale Dataset and Deep Learning-based Quality Metric"
*ACM Transactions on Graphics*

## Questions?

See the repository's issue tracker or contact the project maintainers.
