# QualCompare Revalidation - Quick Start

**Navigation:**
- [Project overview](README.md)
- [Quick metric usage guide](QUICKSTART_METRIC.md) (for custom datasets)

---

This guide walks you through reproducing the QualCompare revalidation results using the pre-trained checkpoint.

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

## Understanding the Results

### Output Files Structure

When evaluating with k-fold checkpoints (`TMQ_NR_8VP_yf03_kfolds`), you'll get:

```
./out/<DATABASE>/<RENDER_METHOD>/<VIEW_METHOD>/<MODEL_NAME>/<N>VP/
├── fold_k0/
│   ├── <reference_objects>/
│   │   └── *_LGLPIPS_scores.csv
│   └── _METRIC_RESULTS_TESTSET_/
│       └── global_combined_correlation.csv
├── fold_k1/ ... fold_k4/ (similar structure)
├── correlation_folds_stats.csv           # Per-fold statistics
└── correlation_summary_kfolds.csv        # Global summary
```

### Interpreting Correlation Metrics

**Two complementary Pearson values are reported:**

1. **Pearson in CSV files (e.g., 0.809)**
   - Calculated as the **mean of per-fold correlations**
   - Each fold's Pearson is computed independently
   - Represents fold-wise performance averaged across splits
   - File: `correlation_folds_stats.csv`

   Example:
   ```
   fold,pearson,spearman
   0,0.7867,0.7791
   1,0.7522,0.7577
   2,0.8027,0.7867
   3,0.8000,0.7990
   4,0.9050,0.9095
   mean,0.8093,0.8064
   ```

2. **Pearson in scatter plot (e.g., 0.803)**
   - Calculated by fitting a **logistic 4PL curve on all folds combined**
   - Measures correlation between the fitted curve and all observations
   - Represents global performance across the entire test set
   - File: `*_ALLFOLDS.png`

**Why the slight difference?**
- CSV values are the average of fold-wise metrics (stable, fold-independent)
- Plot values come from a global fit that optimizes for the entire combined dataset
- Both are valid and complementary views of the metric's performance

**Which to use?**
- Use **CSV values (mean)** for cross-validation reporting and reproducibility
- Use **plot values** for visualizing global trend and quality of fit across all data

### Quality Assessment Guidelines

- **Pearson ≥ 0.80**: Excellent correlation with subjective quality
- **Pearson 0.70-0.80**: Good correlation, suitable for ranking
- **Pearson < 0.70**: Moderate correlation, use with caution

The `TMQ_NR_8VP_yf03_kfolds` checkpoint achieves ~0.81 mean Pearson, indicating strong correlation with subjective MOS scores.

### Reference to Publication Results

These exact results are reported in **Table VI** of:

> Campagne, G., Dupont, F., Lavoué, G., Denis, F., Delanoy, J. (2026)
> *"Towards Reproducible Image-based 3D quality assessment: integrated software and new results"*

When you run the complete revalidation pipeline on the TMQ dataset with the `TMQ_NR_8VP_yf03_kfolds` checkpoint (8 views, Y_fixed_0.3 camera offset, New_Render rendering method), the results you obtain should match the published table:

- **TMQ dataset, 5-fold validation**:
  - Pearson (mean): 0.809
  - Spearman (mean): 0.806

If your results differ significantly, check:
1. That you're using the correct pre-trained checkpoint
2. That the rendered images match the QualCompare output format
3. That the MOS CSV and test list files are properly formatted
4. That `USE_FOLDS=True` is set in the notebook configuration

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

**Original Graphics-LPIPS work:**

Nehme, Y., Delanoy, J., Dupont, F., Farrugia, J., Le Callet, P., Lavoue, G. (2022).
"Textured Mesh Quality Assessment: Large-Scale Dataset and Deep Learning-based Quality Metric"
*ACM Transactions on Graphics*

**QualCompare Revalidation (this workflow):**

Campagne, G., Dupont, F., Lavoué, G., Denis, F., Delanoy, J. (2026).
"Towards Reproducible Image-based 3D quality assessment: integrated software and new results"
*TBD*

**Publication note:** The TMQ dataset results reported in **Table VI** of the 2026 paper are directly reproducible using this repository with the `TMQ_NR_8VP_yf03_kfolds` checkpoint.

## Questions?

See the repository's issue tracker or contact the project maintainers.
