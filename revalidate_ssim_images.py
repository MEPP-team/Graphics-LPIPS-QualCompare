"""Revalidation script for full-view SSIM.

This fixed baseline compares each rendered reference view directly with the
matching distorted view, then writes one SSIM score per viewpoint.
"""

import argparse
import os

import numpy as np

import correlation_VP
import find_dis_ref
from revalidation_common import get_ref_list_for_full_database, has_completed_results_file, load_rgb_image
from ssim import ssim as compute_ssim


RESULTS_FILENAME = "SSIM_IMAGES_results_testset.csv"


def rgb_to_gray(image_rgb):
    return np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", type=str, default="0.1")
    parser.add_argument("-m", "--model", type=str, default="SSIM_IMAGES")
    parser.add_argument("--use_folds", action="store_true", help="accepted for compatibility; ignored for fixed baselines")
    parser.add_argument("-v", "--views", type=int, required=True)
    parser.add_argument("-vm", "--view_method", type=str, required=True)
    parser.add_argument("-rm", "--render_method", type=str, required=True)
    parser.add_argument("-db", "--database", type=str, required=True)
    parser.add_argument("-mos", "--mos_csv_file", type=str, required=True)
    parser.add_argument("-testlist", "--test_list_csv", type=str, default=None)
    parser.add_argument("--src_root", type=str, default=".")
    parser.add_argument("--force", action="store_true", help="recompute result files even if they already exist")
    args = parser.parse_args()

    if args.use_folds:
        print("Ignoring --use_folds: SSIM_IMAGES is a fixed baseline and will be evaluated once on the full database.")

    out = os.path.join(
        ".",
        "out",
        args.database,
        args.render_method,
        args.view_method,
        args.model,
        f"{args.views}VP",
    ) + "/"
    root_ref_views = os.path.join(args.src_root, "Source", f"{args.views}VP")
    root_dis_views = os.path.join(args.src_root, "Distorted", f"{args.views}VP")

    if not os.path.exists(root_ref_views):
        raise FileNotFoundError(f"The folder {root_ref_views} does not exist. Please check the parameters.")
    if not os.path.exists(root_dis_views):
        raise FileNotFoundError(f"The folder {root_dis_views} does not exist. Please check the parameters.")

    ref_obj_list = get_ref_list_for_full_database(root_ref_views, args.test_list_csv)
    print(f"Evaluating full-view SSIM on {len(ref_obj_list)} reference objects.")

    for ref_obj in ref_obj_list:
        ref_views_folder = os.path.join(root_ref_views, ref_obj, "views")
        distorted_obj_list = find_dis_ref.find_dis_files(root_dis_views, ref_obj)

        results_dir = out + "_METRIC_RESULTS_TESTSET_/" + ref_obj + "/"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, RESULTS_FILENAME)

        if has_completed_results_file(results_file) and not args.force:
            print(f"Skipping existing result file {results_file}")
            continue

        print(f"Creating the file {results_file}")
        with open(results_file, "w", newline="") as file_ssim:
            file_ssim.writelines("ObjectName, MOS, SSIM\n")

            for distorted_obj in distorted_obj_list:
                dis_views_folder = os.path.join(root_dis_views, distorted_obj, "views")
                mos_value = correlation_VP.get_MOS(args.mos_csv_file, distorted_obj, name_col=0, mos_col=1)

                view_scores = []
                for view_idx in range(1, args.views + 1):
                    ref_view = load_rgb_image(os.path.join(ref_views_folder, f"view_{view_idx}.png"))
                    dis_view = load_rgb_image(os.path.join(dis_views_folder, f"view_{view_idx}.png"))

                    score, _ = compute_ssim(rgb_to_gray(ref_view), rgb_to_gray(dis_view))
                    view_scores.append(float(score))

                file_ssim.writelines(f"{distorted_obj}, {mos_value:.2f}, ")
                for i, score in enumerate(view_scores):
                    file_ssim.writelines(f"{score:.6f}")
                    if i != len(view_scores) - 1:
                        file_ssim.writelines(", ")
                file_ssim.writelines("\n")

        print(f"Wrote {results_file}")


if __name__ == "__main__":
    main()
