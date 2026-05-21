import csv
import os
import re

import cv2


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"_db$", "", name)
    name = re.sub(r"_kfolds$", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def metric_results_filename(metric_name: str) -> str:
    metric_name = normalize_name(metric_name).upper()
    return f"{metric_name}_results_testset.csv"


def resolve_fold_testlist_path(database: str, test_list_csv: str, fold: int) -> str:
    basename = os.path.basename(test_list_csv)
    basename = re.sub(r"_k\d+(?=\.csv$)", "", basename)
    fold_csv = os.path.join(
        "./dataset",
        database,
        "folds",
        basename.replace(".csv", f"_k{fold}.csv"),
    )

    if os.path.isfile(fold_csv):
        return fold_csv

    if os.path.isfile(test_list_csv):
        return test_list_csv

    raise FileNotFoundError(
        f"Could not find test list CSV for fold {fold}. Tried '{fold_csv}' and '{test_list_csv}'."
    )


def build_fold_testlist_paths(database: str, test_list_csv: str, use_folds: bool):
    if not use_folds:
        return [test_list_csv]

    test_list_fold_paths = []
    for fold in range(5):
        test_list_fold_paths.append(resolve_fold_testlist_path(database, test_list_csv, fold))
    return test_list_fold_paths


def get_rendered_ref_list(root_ref_patches: str):
    return sorted(
        entry
        for entry in os.listdir(root_ref_patches)
        if os.path.isdir(os.path.join(root_ref_patches, entry))
    )


def get_ref_list_for_full_database(root_ref_patches: str, test_list_csv: str | None = None):
    if test_list_csv and os.path.isfile(test_list_csv):
        return get_testset_ref_list(test_list_csv)
    return get_rendered_ref_list(root_ref_patches)


def get_testset_ref_list(test_list_csv):
    ref_list = []
    with open(test_list_csv, mode="r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            name_candidate = row[0]
            if name_candidate not in ref_list:
                ref_list.append(name_candidate)
    return ref_list


def load_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1]


def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def iter_view_patch_groups(csv_patch_file, ref_views_folder, dis_views_folder, ext=".png"):
    with open(csv_patch_file, newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        v = 1
        patch_size = None
        nb_patches_vn = None
        refimg = None
        disimg = None
        patches0 = []
        patches1 = []

        for row in csv_reader:
            if line_count == 0:
                patch_size = int(row[4].split("=")[1].strip())
                nb_patches_vn = [int(r.split("=")[1].strip()) for r in row[7:]]
                refimg = load_rgb_image(f"{ref_views_folder}/view_{v}{ext}")
                disimg = load_rgb_image(f"{dis_views_folder}/view_{v}{ext}")
            else:
                if line_count > sum(nb_patches_vn[0:v]):
                    if patches0:
                        yield v, patch_size, patches0, patches1
                    v += 1
                    refimg = load_rgb_image(f"{ref_views_folder}/view_{v}{ext}")
                    disimg = load_rgb_image(f"{dis_views_folder}/view_{v}{ext}")
                    patches0 = []
                    patches1 = []

                x, y = int(row[0]), int(row[1])
                patch0 = refimg[y : y + patch_size, x : x + patch_size]
                patch1 = disimg[y : y + patch_size, x : x + patch_size]
                if patch0.shape[:2] != (patch_size, patch_size) or patch1.shape[:2] != (patch_size, patch_size):
                    line_count += 1
                    continue
                patches0.append(patch0)
                patches1.append(patch1)

            line_count += 1

        if patches0:
            yield v, patch_size, patches0, patches1
