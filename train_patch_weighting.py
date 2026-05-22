"""Train a weakly supervised patch-weighting model for Graphics-LPIPS scores."""

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch import nn


FEATURE_COLUMNS = [
    "glpips_patch_score",
    "x_center_norm",
    "y_center_norm",
    "view_norm",
    "patch_area_norm",
    "ref_gray_mean",
    "ref_gray_std",
    "ref_gradient_mean",
    "ref_gradient_std",
    "ref_gradient_max",
    "ref_laplacian_var",
    "dis_gray_mean",
    "dis_gray_std",
    "dis_gradient_mean",
    "dis_gradient_std",
    "dis_gradient_max",
    "dis_laplacian_var",
    "diff_gray_mean",
    "diff_gray_std",
    "diff_gradient_mean",
    "diff_gradient_std",
    "diff_gradient_max",
    "diff_laplacian_var",
    "ref_r_mean",
    "ref_r_std",
    "ref_g_mean",
    "ref_g_std",
    "ref_b_mean",
    "ref_b_std",
    "dis_r_mean",
    "dis_r_std",
    "dis_g_mean",
    "dis_g_std",
    "dis_b_mean",
    "dis_b_std",
    "diff_r_mean",
    "diff_r_std",
    "diff_g_mean",
    "diff_g_std",
    "diff_b_mean",
    "diff_b_std",
    "ref_lab_l_mean",
    "ref_lab_l_std",
    "ref_lab_a_mean",
    "ref_lab_a_std",
    "ref_lab_b_mean",
    "ref_lab_b_std",
    "dis_lab_l_mean",
    "dis_lab_l_std",
    "dis_lab_a_mean",
    "dis_lab_a_std",
    "dis_lab_b_mean",
    "dis_lab_b_std",
    "diff_lab_l_mean",
    "diff_lab_l_std",
    "diff_lab_a_mean",
    "diff_lab_a_std",
    "diff_lab_b_mean",
    "diff_lab_b_std",
    "mask_object_ratio",
    "mask_background_ratio",
    "mask_edge_ratio",
    "mask_bbox_area_norm",
    "patch_center_dist_to_object_center",
    "patch_x_in_object_bbox",
    "patch_y_in_object_bbox",
]


class PatchAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def normalize_mos_values(mos_values, higher_better=True):
    mos_values = np.asarray(mos_values, dtype=np.float32)
    mos_min = float(np.min(mos_values))
    mos_max = float(np.max(mos_values))
    if abs(mos_max - mos_min) < 1e-12:
        return np.zeros_like(mos_values), mos_min, mos_max
    normalized = (mos_values - mos_min) / (mos_max - mos_min)
    if higher_better:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32), mos_min, mos_max


def patch_features(row, views):
    score = float(row["glpips_patch_score"])
    x = float(row["x"])
    y = float(row["y"])
    patch_size = float(row["patch_size"])
    view_width = float(row["view_width"])
    view_height = float(row["view_height"])
    view_idx = int(row["view_idx"])

    values = {
        "glpips_patch_score": score,
        "x_center_norm": (x + patch_size / 2.0) / max(view_width, 1.0),
        "y_center_norm": (y + patch_size / 2.0) / max(view_height, 1.0),
        "view_norm": (view_idx - 1.0) / max(views - 1.0, 1.0),
        "patch_area_norm": (patch_size * patch_size) / max(view_width * view_height, 1.0),
    }
    for feature_name in FEATURE_COLUMNS:
        if feature_name not in values:
            if feature_name not in row:
                raise KeyError(
                    f"Missing feature '{feature_name}' in patch-score CSV. "
                    "Regenerate patch scores with the current Light_GraphicsLPIPS_csv.py --save_patch_scores."
                )
            values[feature_name] = float(row[feature_name])

    return [values[feature_name] for feature_name in FEATURE_COLUMNS]


def load_patch_score_groups(patch_scores_root, views):
    groups = []
    for root, _, files in os.walk(patch_scores_root):
        for filename in files:
            if not filename.endswith("_patch_scores.csv"):
                continue
            path = os.path.join(root, filename)
            features = []
            scores = []
            view_indices = []
            ref_obj = None
            distorted_obj = None
            mos = None

            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_obj = row["ref_obj"]
                    distorted_obj = row["distorted_obj"]
                    mos = float(row["mos"])
                    score = float(row["glpips_patch_score"])
                    features.append(patch_features(row, views))
                    scores.append(score)
                    view_indices.append(int(row["view_idx"]))

            if features:
                groups.append(
                    {
                        "path": path,
                        "ref_obj": ref_obj,
                        "distorted_obj": distorted_obj,
                        "mos": mos,
                        "features": np.asarray(features, dtype=np.float32),
                        "scores": np.asarray(scores, dtype=np.float32),
                        "view_indices": np.asarray(view_indices, dtype=np.int64),
                    }
                )

    if not groups:
        raise FileNotFoundError(f"No *_patch_scores.csv files found under {patch_scores_root}")
    return groups


def aggregate_group(model, group, target, feature_mean, feature_std, device):
    features = torch.from_numpy(group["features"]).to(device)
    scores = torch.from_numpy(group["scores"]).to(device)
    view_indices = torch.from_numpy(group["view_indices"]).to(device)

    features = (features - feature_mean) / feature_std
    logits = model(features)

    view_scores = []
    for view_idx in torch.unique(view_indices):
        mask = view_indices == view_idx
        weights = torch.softmax(logits[mask], dim=0)
        view_scores.append(torch.sum(weights * scores[mask]))
    prediction = torch.stack(view_scores).mean()
    loss = (prediction - target) ** 2
    return prediction, loss


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--patch_scores_root", required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--output", default="./checkpoints/WEIGHTED_GLPIPS/patch_attention.pt")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mos_lower_better", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    groups = load_patch_score_groups(args.patch_scores_root, args.views)
    mos_targets, mos_min, mos_max = normalize_mos_values(
        [g["mos"] for g in groups],
        higher_better=not args.mos_lower_better,
    )
    for group, target in zip(groups, mos_targets):
        group["target"] = float(target)

    all_features = np.concatenate([g["features"] for g in groups], axis=0)
    feature_mean_np = all_features.mean(axis=0).astype(np.float32)
    feature_std_np = all_features.std(axis=0).astype(np.float32)
    feature_std_np[feature_std_np < 1e-6] = 1.0

    random.shuffle(groups)
    val_count = max(1, int(round(len(groups) * args.val_fraction))) if len(groups) > 1 else 0
    val_groups = groups[:val_count]
    train_groups = groups[val_count:] if val_count else groups

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model = PatchAttention(input_dim=len(FEATURE_COLUMNS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    feature_mean = torch.from_numpy(feature_mean_np).to(device)
    feature_std = torch.from_numpy(feature_std_np).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_groups)
        train_losses = []
        for group in train_groups:
            target = torch.tensor(group["target"], dtype=torch.float32, device=device)
            _, loss = aggregate_group(model, group, target, feature_mean, feature_std, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for group in val_groups:
                target = torch.tensor(group["target"], dtype=torch.float32, device=device)
                _, loss = aggregate_group(model, group, target, feature_mean, feature_std, device)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch {epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": args.hidden_dim,
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": feature_mean_np,
            "feature_std": feature_std_np,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "mos_lower_better": args.mos_lower_better,
            "views": args.views,
        },
        args.output,
    )
    print(f"Saved patch-weighting model to {args.output}")


if __name__ == "__main__":
    main()
