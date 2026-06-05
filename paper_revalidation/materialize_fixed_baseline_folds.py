"""Filter full fixed-baseline results into the five dataset test folds."""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_revalidation.revalidation_common import normalize_name, resolve_fold_testlist_path


def load_fold_members(path):
    members = {}
    with open(path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            ref_name = normalize_name(row[0])
            members.setdefault(ref_name, set()).add(normalize_name(row[1]))
    return members


def filter_results(source, destination, allowed_distortions):
    with open(source, newline="") as source_file:
        rows = list(csv.reader(source_file))
    if not rows:
        raise ValueError(f"Empty metric result file: {source}")

    filtered = [rows[0]]
    filtered.extend(
        row for row in rows[1:]
        if row and normalize_name(row[0]) in allowed_distortions
    )

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", newline="") as destination_file:
        csv.writer(destination_file).writerows(filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--views", required=True, type=int)
    parser.add_argument("--render-method", required=True)
    parser.add_argument("--view-method", required=True)
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--out-root", default="./out")
    args = parser.parse_args()

    experiment_dir = Path(args.out_root) / args.database / args.render_method / args.view_method / args.model / f"{args.views}VP"
    full_results = experiment_dir / "_METRIC_RESULTS_TESTSET_"

    if not full_results.is_dir():
        raise FileNotFoundError(f"Missing full fixed-baseline results: {full_results}")

    for fold in range(5):
        fold_testlist = resolve_fold_testlist_path(args.database, args.testlist, fold)
        members = load_fold_members(fold_testlist)
        fold_results = experiment_dir / f"fold_k{fold}" / "_METRIC_RESULTS_TESTSET_"
        if fold_results.exists():
            shutil.rmtree(fold_results)

        written = 0
        for object_dir in full_results.iterdir():
            if not object_dir.is_dir():
                continue
            allowed = members.get(normalize_name(object_dir.name))
            if not allowed:
                continue
            source = object_dir / args.results_file
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = fold_results / object_dir.name / args.results_file
            filter_results(source, destination, allowed)
            written += 1

        if not written:
            raise RuntimeError(f"Fold {fold} produced no result files from {fold_testlist}")
        print(f"Fold {fold}: wrote {written} reference-object result files from {fold_testlist}")


if __name__ == "__main__":
    main()
