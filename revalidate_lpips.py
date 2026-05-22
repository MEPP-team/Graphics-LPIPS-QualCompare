"""Revalidation script for LPIPS using the same rendered-patch flow as GraphicsLPIPS.

LPIPS is used here as a fixed baseline, so it is evaluated once on the full
rendered database rather than fold by fold.
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import correlation_VP
import find_dis_ref
from revalidation_common import (
    get_ref_list_for_full_database,
    has_completed_results_file,
    iter_view_patch_groups,
    metric_results_filename,
)


def im2tensor_01_to_m11(image_rgb):
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", type=str, default="0.1")
    parser.add_argument("-m", "--model", type=str, default="LPIPS_TORCHMETRICS")
    parser.add_argument("--use_folds", action="store_true", help="accepted for compatibility; ignored for fixed baselines")
    parser.add_argument("-v", "--views", type=int, required=True)
    parser.add_argument("-vm", "--view_method", type=str, required=True)
    parser.add_argument("-rm", "--render_method", type=str, required=True)
    parser.add_argument("-db", "--database", type=str, required=True)
    parser.add_argument("-mos", "--mos_csv_file", type=str, required=True)
    parser.add_argument("-testlist", "--test_list_csv", type=str, default=None)
    parser.add_argument("--src_root", type=str, default=".")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="recompute result files even if they already exist")
    args = parser.parse_args()

    model = args.model
    testing_views = args.views
    view_method = args.view_method
    render_method = args.render_method
    database = args.database
    mos_csv_file = args.mos_csv_file
    src_root = args.src_root
    if args.use_folds:
        print("Ignoring --use_folds: LPIPS is a fixed baseline and will be evaluated once on the full database.")

    out = os.path.join(".", "out", database, render_method, view_method, model, f"{testing_views}VP") + "/"
    root_refPatches = os.path.join(src_root, "Source", f"{testing_views}VP")
    root_disPatches = os.path.join(src_root, "Distorted", f"{testing_views}VP")

    if not os.path.exists(root_refPatches):
        raise FileNotFoundError(f"The folder {root_refPatches} does not exist. Please check the parameters.")
    if not os.path.exists(root_disPatches):
        raise FileNotFoundError(f"The folder {root_disPatches} does not exist. Please check the parameters.")

    ext = ".png"
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        reduction="none",
        normalize=False,
    ).to(device)

    ref_obj_list = get_ref_list_for_full_database(root_refPatches, args.test_list_csv)
    output_dir = out
    print(f"Evaluating LPIPS on {len(ref_obj_list)} reference objects.")

    if args.use_gpu and torch.cuda.is_available():
        print("Using GPU for LPIPS evaluation.")

    for ref_obj in ref_obj_list:
        ref_obj_root = os.path.join(root_refPatches, ref_obj)
        ref_views_folder = os.path.join(ref_obj_root, "views")
        distorted_obj_list = find_dis_ref.find_dis_files(root_disPatches, ref_obj)

        results_dir = output_dir + "_METRIC_RESULTS_TESTSET_/" + ref_obj + "/"
        os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        results_file = results_dir + metric_results_filename("LPIPS")

        if has_completed_results_file(results_file) and not args.force:
            print(f"Skipping existing result file {results_file}")
            continue

        print(f"Creating the file {results_file}")
        with open(results_file, "w", newline="") as file_lpips:
            file_lpips.writelines("ObjectName, MOS, LPIPS\n")

            for distorted_obj in distorted_obj_list:
                dis_views_folder = os.path.join(root_disPatches, distorted_obj, "views")
                csv_patch_files = find_dis_ref.find_ref_csvfiles(ref_obj_root)
                if not csv_patch_files:
                    raise FileNotFoundError(
                        f"No patch CSV found under {ref_obj_root}. Expected a reference patchlist CSV in the QualCompare output tree."
                    )
                csv_patch_file = csv_patch_files[0]
                mos_value = correlation_VP.get_MOS(mos_csv_file, distorted_obj, name_col=0, mos_col=1)

                list_graphics_lpips = []
                for _, patch_size, patches0, patches1 in iter_view_patch_groups(
                    csv_patch_file,
                    ref_views_folder,
                    dis_views_folder,
                    ext=ext,
                ):
                    batch0 = torch.stack([im2tensor_01_to_m11(p) for p in patches0], dim=0).to(device)
                    batch1 = torch.stack([im2tensor_01_to_m11(p) for p in patches1], dim=0).to(device)

                    with torch.no_grad():
                        dists = metric(batch0, batch1).view(-1).detach().cpu().numpy()
                    np.clip(dists, 0.0, 1.0, out=dists)
                    list_graphics_lpips.append(float(dists.mean()))

                file_lpips.writelines(f"{distorted_obj}, {mos_value:.2f}, ")
                for i, score in enumerate(list_graphics_lpips):
                    file_lpips.writelines(f"{score:.6f}")
                    if i != len(list_graphics_lpips) - 1:
                        file_lpips.writelines(", ")
                file_lpips.writelines("\n")

        print(f"Wrote {results_file}")


if __name__ == "__main__":
    main()
