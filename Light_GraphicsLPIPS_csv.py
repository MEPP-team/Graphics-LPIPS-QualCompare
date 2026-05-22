"""Evaluate Graphics-LPIPS on rendered views by reconstructing patches in memory."""

import argparse
import csv
import os
import re

import cv2
import lpips
import numpy as np
import torch

import correlation_VP
import find_dis_ref


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"_db$", "", name)
    name = re.sub(r"_kfolds$", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def rgb_to_gray_float(image_rgb):
    return np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def gradient_mean(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(grad_x * grad_x + grad_y * grad_y).mean() / 255.0)


def gradient_stats(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y) / 255.0
    return {
        "mean": float(magnitude.mean()),
        "std": float(magnitude.std()),
        "max": float(magnitude.max()),
    }


def laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var() / (255.0 * 255.0))


def rgb_channel_stats(prefix, patch):
    patch_float = patch.astype(np.float32) / 255.0
    features = {}
    for idx, channel in enumerate(["r", "g", "b"]):
        channel_values = patch_float[..., idx]
        features[f"{prefix}_{channel}_mean"] = float(channel_values.mean())
        features[f"{prefix}_{channel}_std"] = float(channel_values.std())
    return features


def lab_channel_stats(prefix, patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] /= 255.0
    lab[..., 1] /= 255.0
    lab[..., 2] /= 255.0
    features = {}
    for idx, channel in enumerate(["l", "a", "b"]):
        channel_values = lab[..., idx]
        features[f"{prefix}_lab_{channel}_mean"] = float(channel_values.mean())
        features[f"{prefix}_lab_{channel}_std"] = float(channel_values.std())
    return features


def load_mask_context(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    binary = mask > 127
    if not np.any(binary):
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    ys, xs = np.where(binary)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    height, width = binary.shape
    return {
        "binary": binary,
        "edge": edge,
        "bbox": (x_min, y_min, x_max, y_max),
        "center": ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
        "size": (width, height),
    }


def patch_mask_features(mask_context, x, y, patch_size):
    if mask_context is None:
        return {
            "mask_object_ratio": 0.0,
            "mask_background_ratio": 1.0,
            "mask_edge_ratio": 0.0,
            "mask_bbox_area_norm": 0.0,
            "patch_center_dist_to_object_center": 1.0,
            "patch_x_in_object_bbox": 0.0,
            "patch_y_in_object_bbox": 0.0,
        }

    binary = mask_context["binary"]
    edge = mask_context["edge"]
    view_width, view_height = mask_context["size"]
    x_min, y_min, x_max, y_max = mask_context["bbox"]
    center_x, center_y = mask_context["center"]

    patch_binary = binary[y : y + patch_size, x : x + patch_size]
    patch_edge = edge[y : y + patch_size, x : x + patch_size]
    object_ratio = float(patch_binary.mean()) if patch_binary.size else 0.0
    edge_ratio = float(patch_edge.mean()) if patch_edge.size else 0.0

    patch_center_x = x + patch_size / 2.0
    patch_center_y = y + patch_size / 2.0
    diagonal = max(float(np.sqrt(view_width * view_width + view_height * view_height)), 1.0)
    dist_to_center = float(np.sqrt((patch_center_x - center_x) ** 2 + (patch_center_y - center_y) ** 2) / diagonal)

    bbox_width = max(float(x_max - x_min + 1), 1.0)
    bbox_height = max(float(y_max - y_min + 1), 1.0)
    bbox_area_norm = float((bbox_width * bbox_height) / max(view_width * view_height, 1.0))
    x_in_bbox = float((patch_center_x - x_min) / bbox_width)
    y_in_bbox = float((patch_center_y - y_min) / bbox_height)

    return {
        "mask_object_ratio": object_ratio,
        "mask_background_ratio": 1.0 - object_ratio,
        "mask_edge_ratio": edge_ratio,
        "mask_bbox_area_norm": bbox_area_norm,
        "patch_center_dist_to_object_center": dist_to_center,
        "patch_x_in_object_bbox": x_in_bbox,
        "patch_y_in_object_bbox": y_in_bbox,
    }


def patch_quality_features(ref_patch, dis_patch, mask_context, x, y, patch_size):
    ref_gray = rgb_to_gray_float(ref_patch)
    dis_gray = rgb_to_gray_float(dis_patch)
    diff_gray = np.abs(ref_gray - dis_gray)
    ref_gradient = gradient_stats(ref_gray)
    dis_gradient = gradient_stats(dis_gray)
    diff_gradient = gradient_stats(diff_gray)
    diff_rgb = np.abs(ref_patch.astype(np.float32) - dis_patch.astype(np.float32)).astype(np.uint8)

    features = {
        "ref_gray_mean": float(ref_gray.mean() / 255.0),
        "ref_gray_std": float(ref_gray.std() / 255.0),
        "ref_gradient_mean": ref_gradient["mean"],
        "ref_gradient_std": ref_gradient["std"],
        "ref_gradient_max": ref_gradient["max"],
        "ref_laplacian_var": laplacian_variance(ref_gray),
        "dis_gray_mean": float(dis_gray.mean() / 255.0),
        "dis_gray_std": float(dis_gray.std() / 255.0),
        "dis_gradient_mean": dis_gradient["mean"],
        "dis_gradient_std": dis_gradient["std"],
        "dis_gradient_max": dis_gradient["max"],
        "dis_laplacian_var": laplacian_variance(dis_gray),
        "diff_gray_mean": float(diff_gray.mean() / 255.0),
        "diff_gray_std": float(diff_gray.std() / 255.0),
        "diff_gradient_mean": diff_gradient["mean"],
        "diff_gradient_std": diff_gradient["std"],
        "diff_gradient_max": diff_gradient["max"],
        "diff_laplacian_var": laplacian_variance(diff_gray),
    }
    features.update(rgb_channel_stats("ref", ref_patch))
    features.update(rgb_channel_stats("dis", dis_patch))
    features.update(rgb_channel_stats("diff", diff_rgb))
    features.update(lab_channel_stats("ref", ref_patch))
    features.update(lab_channel_stats("dis", dis_patch))
    features.update(lab_channel_stats("diff", diff_rgb))
    features.update(patch_mask_features(mask_context, x, y, patch_size))
    return features


PATCH_FEATURE_COLUMNS = [
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


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--use_gpu", action="store_true", default=True, help="turn on flag to use GPU")
parser.add_argument("--version", type=str, default="0.1")
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--use_folds", action="store_true")
parser.add_argument("-v", "--views", type=int, required=True)
parser.add_argument("-vm", "--view_method", type=str, required=True)
parser.add_argument("-rm", "--render_method", type=str, required=True)
parser.add_argument("-db", "--database", type=str, required=True)
parser.add_argument("-mos", "--mos_csv_file", type=str, required=True)
parser.add_argument("-testlist", "--test_list_csv", type=str, required=True)
parser.add_argument("--src_root", type=str, default=".", help="root directory containing Source/ and Distorted/ experiment folders")
parser.add_argument("--save_patch_scores", action="store_true", help="save one Graphics-LPIPS score per reconstructed patch")
opt = parser.parse_args()

model = opt.model
modelpath = "./checkpoints/" + model + "/latest_net_.pth"
use_folds = opt.use_folds
testing_views = opt.views
view_method = opt.view_method
render_method = opt.render_method
database = opt.database
mos_csv_file = opt.mos_csv_file
test_list_csv = opt.test_list_csv
src_root = opt.src_root
force_overwrite = False
out = os.path.join(".", "out", database, render_method, view_method, model, str(testing_views) + "VP") + "/"

root_refPatches = os.path.join(src_root, "Source", str(testing_views) + "VP")
if not os.path.exists(root_refPatches):
    print("The folder %s does not exist. Please check the parameters." % root_refPatches)
    exit()

root_disPatches = os.path.join(src_root, "Distorted", str(testing_views) + "VP")
if not os.path.exists(root_disPatches):
    print("The folder %s does not exist. Please check the parameters." % root_disPatches)
    exit()

ext = ".png"

if use_folds:
    ref_obj_list_folds = []
    model_folds = []
    output_folds = []
    for fold in range(5):
        model_norm = normalize_name(model)
        db_norm = normalize_name(database)
        if model_norm.startswith(db_norm) or (model_norm.startswith("graphicslpips") and db_norm == "tmq"):
            test_list_csv_fold = (
                "./dataset/" + database + "/folds/" + os.path.basename(test_list_csv).replace(".csv", f"_k{fold}.csv")
            )
        else:
            print("Warning: The model name does not match the database name. We will use the same test list CSV file for all folds.")
            test_list_csv_fold = test_list_csv

        print("Using test list CSV file for fold %d: %s" % (fold, test_list_csv_fold))
        ref_obj_list_folds.append(correlation_VP.get_testset_ref_list(test_list_csv_fold))
        model_folds.append("./checkpoints/" + model + "/fold_k" + str(fold) + "/latest_net_.pth")
        output_folds.append(out + "fold_k" + str(fold) + "/")
else:
    ref_obj_list_folds = [correlation_VP.get_testset_ref_list(test_list_csv)]
    model_folds = [modelpath]
    print("Evaluating the test set with %s model" % model)
    output_folds = [out]


List_MOS = []
for fold_idx, ref_obj_list in enumerate(ref_obj_list_folds):
    loss_fn = lpips.LPIPS(net="alex", version=opt.version, model_path=model_folds[fold_idx])
    if opt.use_gpu:
        loss_fn.cuda()
        print("Using GPU for evaluation.")

    sd = loss_fn.state_dict()
    print("CKPT loaded keys:", len(sd))
    for k in ["lins.0.model.1.weight", "net.slice1.0.weight"]:
        if k in sd:
            print(k, float(sd[k].abs().sum()))

    if not os.path.exists(output_folds[fold_idx]):
        os.makedirs(os.path.dirname(output_folds[fold_idx]), exist_ok=True)

    if use_folds:
        print("--- Starting fold k%d ---" % fold_idx)

    for ref_obj in ref_obj_list:
        ref_obj_root = os.path.join(root_refPatches, ref_obj)
        ref_views_folder = os.path.join(ref_obj_root, "views")
        ref_masks_folder = os.path.join(ref_obj_root, "masks")
        distorted_obj_list = find_dis_ref.find_dis_files(root_disPatches, ref_obj)
        currentFolder = output_folds[fold_idx] + ref_obj + "/"

        results_dir = output_folds[fold_idx] + "_METRIC_RESULTS_TESTSET_/" + ref_obj + "/"
        if not os.path.exists(results_dir):
            os.makedirs(os.path.dirname(results_dir), exist_ok=True)

        results_file = results_dir + "GLPIPS_results_testset.csv"
        if os.path.exists(results_file) and force_overwrite is False and not opt.save_patch_scores:
            print("The file %s already exists. We will not overwrite it." % results_file)
            continue

        print("Creating the file %s" % results_file)
        file_GLPIPS = open(results_file, "w")
        file_GLPIPS.writelines("ObjectName, MOS, LPIPS\n")

        for distorted_obj in distorted_obj_list:
            List_GraphicsLPIPS = []
            outcsvfile = currentFolder + distorted_obj + "_LGLPIPS_scores.csv"

            dis_views_folder = os.path.join(root_disPatches, distorted_obj, "views")
            csv_patch_files = find_dis_ref.find_ref_csvfiles(ref_obj_root)
            if not csv_patch_files:
                raise FileNotFoundError(
                    f"No patch CSV found under {ref_obj_root}. Expected a reference patchlist CSV in the QualCompare output tree."
                )
            csv_patch_file = csv_patch_files[0]
            mos_value = correlation_VP.get_MOS(mos_csv_file, distorted_obj, name_col=0, mos_col=1)
            List_MOS.append([mos_value])

            patch_score_writer = None
            patch_score_file = None
            if opt.save_patch_scores:
                patch_scores_dir = output_folds[fold_idx] + "_PATCH_SCORES_TESTSET_/" + ref_obj + "/"
                os.makedirs(patch_scores_dir, exist_ok=True)
                patch_scores_path = os.path.join(patch_scores_dir, distorted_obj + "_patch_scores.csv")
                patch_score_file = open(patch_scores_path, "w", newline="")
                patch_score_writer = csv.writer(patch_score_file)
                patch_score_writer.writerow(
                    [
                        "ref_obj",
                        "distorted_obj",
                        "mos",
                        "view_idx",
                        "patch_idx",
                        "x",
                        "y",
                        "patch_size",
                        "view_width",
                        "view_height",
                        "glpips_patch_score",
                    ]
                    + PATCH_FEATURE_COLUMNS
                )

            def write_patch_scores(view_idx, patch_size, ref_image, patch_meta, dists):
                if patch_score_writer is None:
                    return
                view_height, view_width = ref_image.shape[:2]
                for meta, score in zip(patch_meta, dists):
                    patch_score_writer.writerow(
                        [
                            ref_obj,
                            distorted_obj,
                            "%.6f" % mos_value,
                            view_idx,
                            meta["patch_idx"],
                            meta["x"],
                            meta["y"],
                            patch_size,
                            view_width,
                            view_height,
                            "%.8f" % float(score),
                        ]
                        + ["%.8f" % meta[column] for column in PATCH_FEATURE_COLUMNS]
                    )

            with open(csv_patch_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                line_count = 0
                v = 1

                for row in csv_reader:
                    if line_count == 0:
                        patchSize = int(row[4].split("=")[1].strip())
                        nbPatchesVn = [int(r.split("=")[1].strip()) for r in row[7:]]

                        refimg = cv2.imread(f"{ref_views_folder}/view_{v}{ext}")[:, :, ::-1]
                        disimg = cv2.imread(f"{dis_views_folder}/view_{v}{ext}")[:, :, ::-1]
                        mask_context = load_mask_context(f"{ref_masks_folder}/mask_{v}{ext}")
                        patches0 = []
                        patches1 = []
                        patch_meta = []
                        patch_idx = 0
                    else:
                        if line_count > sum(nbPatchesVn[0:v]):
                            if patches0:
                                batch0 = torch.cat([lpips.im2tensor(p).cuda() for p in patches0], dim=0)
                                batch1 = torch.cat([lpips.im2tensor(p).cuda() for p in patches1], dim=0)
                                with torch.no_grad():
                                    dists_t = loss_fn(batch0, batch1).view(-1)
                                    dists_np = dists_t.detach().cpu().numpy()
                                    np.clip(dists_np, 0.0, 1.0, out=dists_np)
                                List_GraphicsLPIPS.append(float(dists_np.mean()))
                                write_patch_scores(v, patchSize, refimg, patch_meta, dists_np)

                            v += 1
                            refimg = cv2.imread(f"{ref_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            disimg = cv2.imread(f"{dis_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            mask_context = load_mask_context(f"{ref_masks_folder}/mask_{v}{ext}")
                            patches0 = []
                            patches1 = []
                            patch_meta = []
                            patch_idx = 0

                        x, y = int(row[0]), int(row[1])
                        patch0 = refimg[y : y + patchSize, x : x + patchSize]
                        patch1 = disimg[y : y + patchSize, x : x + patchSize]
                        if patch0.shape[:2] != (patchSize, patchSize) or patch1.shape[:2] != (patchSize, patchSize):
                            continue
                        patches0.append(patch0)
                        patches1.append(patch1)
                        meta = {"patch_idx": patch_idx, "x": x, "y": y}
                        if opt.save_patch_scores:
                            meta.update(patch_quality_features(patch0, patch1, mask_context, x, y, patchSize))
                        patch_meta.append(meta)
                        patch_idx += 1

                    line_count += 1

                if patches0:
                    batch0 = torch.cat([lpips.im2tensor(p).cuda() for p in patches0], dim=0)
                    batch1 = torch.cat([lpips.im2tensor(p).cuda() for p in patches1], dim=0)
                    with torch.no_grad():
                        dists = loss_fn(batch0, batch1).view(-1).cpu().numpy()
                        np.clip(dists, 0.0, 1.0, out=dists)
                    List_GraphicsLPIPS.append(dists.mean())
                    write_patch_scores(v, patchSize, refimg, patch_meta, dists)

            if patch_score_file is not None:
                patch_score_file.close()

            List_MOS[-1].append(List_GraphicsLPIPS)
            file_GLPIPS.writelines("%s, %.2f, " % (distorted_obj, List_MOS[-1][0]))
            for i in range(len(List_GraphicsLPIPS)):
                file_GLPIPS.writelines("%.6f" % List_GraphicsLPIPS[i])
                if i != len(List_GraphicsLPIPS) - 1:
                    file_GLPIPS.writelines(", ")
            file_GLPIPS.writelines("\n")
        file_GLPIPS.close()
