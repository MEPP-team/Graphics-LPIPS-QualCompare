"""Revalidation script for SSIM using the same rendered-patch flow as GraphicsLPIPS.

SSIM is a fixed baseline, so it is evaluated once on the full rendered database
rather than fold by fold.
"""

import argparse
import os

import numpy as np

import correlation_VP
import find_dis_ref
from revalidation_common import get_ref_list_for_full_database, iter_view_patch_groups, metric_results_filename
from ssim import ssim as compute_ssim


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", type=str, default="0.1")
    parser.add_argument("-m", "--model", type=str, default="SSIM")
    parser.add_argument("--use_folds", action="store_true", help="accepted for compatibility; ignored for fixed baselines")
    parser.add_argument("-v", "--views", type=int, required=True)
    parser.add_argument("-vm", "--view_method", type=str, required=True)
    parser.add_argument("-rm", "--render_method", type=str, required=True)
    parser.add_argument("-db", "--database", type=str, required=True)
    parser.add_argument("-mos", "--mos_csv_file", type=str, required=True)
    parser.add_argument("-testlist", "--test_list_csv", type=str, default=None)
    parser.add_argument("--src_root", type=str, default=".")
    args = parser.parse_args()

    model = args.model
    testing_views = args.views
    view_method = args.view_method
    render_method = args.render_method
    database = args.database
    mos_csv_file = args.mos_csv_file
    src_root = args.src_root
    if args.use_folds:
        print("Ignoring --use_folds: SSIM is a fixed baseline and will be evaluated once on the full database.")

    out = os.path.join(".", "out", database, render_method, view_method, model, f"{testing_views}VP") + "/"
    root_refPatches = os.path.join(src_root, "Source", f"{testing_views}VP")
    root_disPatches = os.path.join(src_root, "Distorted", f"{testing_views}VP")

    if not os.path.exists(root_refPatches):
        raise FileNotFoundError(f"The folder {root_refPatches} does not exist. Please check the parameters.")
    if not os.path.exists(root_disPatches):
        raise FileNotFoundError(f"The folder {root_disPatches} does not exist. Please check the parameters.")

    ext = ".png"

    ref_obj_list = get_ref_list_for_full_database(root_refPatches, args.test_list_csv)
    output_dir = out
    print(f"Evaluating SSIM on {len(ref_obj_list)} reference objects.")

    for ref_obj in ref_obj_list:
        ref_obj_root = os.path.join(root_refPatches, ref_obj)
        ref_views_folder = os.path.join(ref_obj_root, "views")
        distorted_obj_list = find_dis_ref.find_dis_files(root_disPatches, ref_obj)

        results_dir = output_dir + "_METRIC_RESULTS_TESTSET_/" + ref_obj + "/"
        os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        results_file = results_dir + metric_results_filename("SSIM")

        print(f"Creating the file {results_file}")
        with open(results_file, "w", newline="") as file_ssim:
            file_ssim.writelines("ObjectName, MOS, SSIM\n")

            for distorted_obj in distorted_obj_list:
                dis_views_folder = os.path.join(root_disPatches, distorted_obj, "views")
                csv_patch_files = find_dis_ref.find_ref_csvfiles(ref_obj_root)
                if not csv_patch_files:
                    raise FileNotFoundError(
                        f"No patch CSV found under {ref_obj_root}. Expected a reference patchlist CSV in the QualCompare output tree."
                    )
                csv_patch_file = csv_patch_files[0]
                mos_value = correlation_VP.get_MOS(mos_csv_file, distorted_obj, name_col=0, mos_col=1)

                list_ssim = []
                for _, _, patches0, patches1 in iter_view_patch_groups(
                    csv_patch_file,
                    ref_views_folder,
                    dis_views_folder,
                    ext=ext,
                ):
                    patch_scores = []
                    for patch0, patch1 in zip(patches0, patches1):
                        gray0 = np.dot(patch0[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        gray1 = np.dot(patch1[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        score, _ = compute_ssim(gray0, gray1)
                        patch_scores.append(float(score))

                    list_ssim.append(float(np.mean(patch_scores)))

                file_ssim.writelines(f"{distorted_obj}, {mos_value:.2f}, ")
                for i, score in enumerate(list_ssim):
                    file_ssim.writelines(f"{score:.6f}")
                    if i != len(list_ssim) - 1:
                        file_ssim.writelines(", ")
                file_ssim.writelines("\n")

        print(f"Wrote {results_file}")


if __name__ == "__main__":
    main()
