"""Apply a trained patch-weighting model to exported Graphics-LPIPS patch scores."""

import argparse
import csv
import os

import numpy as np
import torch

from train_patch_weighting import FEATURE_COLUMNS, PatchAttention, patch_features


RESULTS_FILENAME = "WEIGHTED_GLPIPS_results_testset.csv"


def load_attention_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = PatchAttention(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32, device=device)
    return model, feature_mean, feature_std, checkpoint


def weighted_view_scores(rows, model, feature_mean, feature_std, views, device):
    features = np.asarray([patch_features(row, views) for row in rows], dtype=np.float32)
    scores = np.asarray([float(row["glpips_patch_score"]) for row in rows], dtype=np.float32)
    view_indices = np.asarray([int(row["view_idx"]) for row in rows], dtype=np.int64)

    with torch.no_grad():
        features_t = torch.from_numpy(features).to(device)
        scores_t = torch.from_numpy(scores).to(device)
        view_indices_t = torch.from_numpy(view_indices).to(device)
        logits = model((features_t - feature_mean) / feature_std)

        output_scores = []
        for view_idx in range(1, views + 1):
            mask = view_indices_t == view_idx
            if not bool(torch.any(mask)):
                output_scores.append(float("nan"))
                continue
            weights = torch.softmax(logits[mask], dim=0)
            score = torch.sum(weights * scores_t[mask])
            output_scores.append(float(score.detach().cpu()))
    return output_scores


def iter_patch_score_files(patch_scores_root):
    for root, _, files in os.walk(patch_scores_root):
        for filename in files:
            if filename.endswith("_patch_scores.csv"):
                yield os.path.join(root, filename)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default="WEIGHTED_GLPIPS")
    parser.add_argument("--source_model", type=str, required=True)
    parser.add_argument("--attention_model", type=str, required=True)
    parser.add_argument("--patch_scores_root", type=str, default=None)
    parser.add_argument("-v", "--views", type=int, required=True)
    parser.add_argument("-vm", "--view_method", type=str, required=True)
    parser.add_argument("-rm", "--render_method", type=str, required=True)
    parser.add_argument("-db", "--database", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="./out")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    patch_scores_root = args.patch_scores_root
    if patch_scores_root is None:
        patch_scores_root = os.path.join(
            args.out_root,
            args.database,
            args.render_method,
            args.view_method,
            args.source_model,
            f"{args.views}VP",
            "_PATCH_SCORES_TESTSET_",
        )
    if not os.path.isdir(patch_scores_root):
        raise FileNotFoundError(f"Patch score directory not found: {patch_scores_root}")

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model, feature_mean, feature_std, checkpoint = load_attention_model(args.attention_model, device)
    if checkpoint.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The attention model feature columns do not match this script.")

    result_root = os.path.join(
        args.out_root,
        args.database,
        args.render_method,
        args.view_method,
        args.model,
        f"{args.views}VP",
        "_METRIC_RESULTS_TESTSET_",
    )

    rows_by_ref = {}
    for patch_file in iter_patch_score_files(patch_scores_root):
        with open(patch_file, newline="") as in_file:
            rows = list(csv.DictReader(in_file))
        if not rows:
            continue
        ref_obj = rows[0]["ref_obj"]
        distorted_obj = rows[0]["distorted_obj"]
        rows_by_ref.setdefault(ref_obj, {})[distorted_obj] = rows

    total_rows = 0
    for ref_obj, distorted_rows in sorted(rows_by_ref.items()):
        result_dir = os.path.join(result_root, ref_obj)
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, RESULTS_FILENAME)

        with open(result_file, "w", newline="") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["ObjectName", "MOS", "LPIPS"])

            for distorted_obj, rows in sorted(distorted_rows.items()):
                mos = float(rows[0]["mos"])
                view_scores = weighted_view_scores(rows, model, feature_mean, feature_std, args.views, device)
                writer.writerow([distorted_obj, f"{mos:.2f}", *[f"{score:.6f}" for score in view_scores]])
                total_rows += 1

        print(f"Wrote {result_file}")

    print(f"Wrote {total_rows} weighted distorted-object rows.")


if __name__ == "__main__":
    main()
