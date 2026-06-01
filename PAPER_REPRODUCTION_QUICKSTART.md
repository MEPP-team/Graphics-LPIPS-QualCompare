# QualCompare Revalidation Quick Start

The maintained paper revalidation documentation is now in
[paper_revalidation/README.md](paper_revalidation/README.md).

Use this quick path when you already have QualCompare renders:

```cmd
set QUALCOMPARE_OUT_ROOT=D:\path\to\QualCompare\out
paper_revalidation\revalidate_table_qualcompare.bat --dry-run --preset WPC_SP_CIRCLE_5FOLD
paper_revalidation\revalidate_table_qualcompare.bat --preset WPC_SP_CIRCLE_5FOLD
```

For fixed baselines:

```cmd
set QUALCOMPARE_OUT_ROOT=D:\path\to\QualCompare\out
paper_revalidation\revalidate_fixed_baselines_qualcompare.bat --dry-run --preset WPC_SP_CIRCLE
paper_revalidation\revalidate_fixed_baselines_qualcompare.bat --preset WPC_SP_CIRCLE
```

Outputs are written under:

```text
out/<DATABASE>/<RENDER_METHOD>/<VIEW_METHOD>/<MODEL>/<N>VP/
```

See [paper_revalidation/README.md](paper_revalidation/README.md) for the full
A-to-Z pipeline, required inputs, presets, and troubleshooting notes.
