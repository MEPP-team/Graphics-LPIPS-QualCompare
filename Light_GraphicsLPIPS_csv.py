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
        distorted_obj_list = find_dis_ref.find_dis_files(root_disPatches, ref_obj)
        currentFolder = output_folds[fold_idx] + ref_obj + "/"

        results_dir = output_folds[fold_idx] + "_METRIC_RESULTS_TESTSET_/" + ref_obj + "/"
        if not os.path.exists(results_dir):
            os.makedirs(os.path.dirname(results_dir), exist_ok=True)

        results_file = results_dir + "GLPIPS_results_testset.csv"
        if os.path.exists(results_file) and force_overwrite is False:
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
            List_MOS.append([correlation_VP.get_MOS(mos_csv_file, distorted_obj, name_col=0, mos_col=1)])

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
                        patches0 = []
                        patches1 = []
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

                            v += 1
                            refimg = cv2.imread(f"{ref_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            disimg = cv2.imread(f"{dis_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            patches0 = []
                            patches1 = []

                        x, y = int(row[0]), int(row[1])
                        patch0 = refimg[y : y + patchSize, x : x + patchSize]
                        patch1 = disimg[y : y + patchSize, x : x + patchSize]
                        if patch0.shape[:2] != (patchSize, patchSize) or patch1.shape[:2] != (patchSize, patchSize):
                            continue
                        patches0.append(patch0)
                        patches1.append(patch1)

                    line_count += 1

                if patches0:
                    batch0 = torch.cat([lpips.im2tensor(p).cuda() for p in patches0], dim=0)
                    batch1 = torch.cat([lpips.im2tensor(p).cuda() for p in patches1], dim=0)
                    with torch.no_grad():
                        dists = loss_fn(batch0, batch1).view(-1).cpu().numpy()
                        np.clip(dists, 0.0, 1.0, out=dists)
                    List_GraphicsLPIPS.append(dists.mean())

            List_MOS[-1].append(List_GraphicsLPIPS)
            file_GLPIPS.writelines("%s, %.2f, " % (distorted_obj, List_MOS[-1][0]))
            for i in range(len(List_GraphicsLPIPS)):
                file_GLPIPS.writelines("%.6f" % List_GraphicsLPIPS[i])
                if i != len(List_GraphicsLPIPS) - 1:
                    file_GLPIPS.writelines(", ")
            file_GLPIPS.writelines("\n")
        file_GLPIPS.close()
