import numpy as np
import h5py
# from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5, check_dir_and_create, writeHDF5attribute, parse_data_slice
from segmfriends.utils import various as var_utils
from segmfriends.utils import segm_utils as segmutils

import vigra
import pandas as pd
import scipy.ndimage
import cv2

from sklearn.metrics import confusion_matrix

# ----------------
# VARIOUS FOLDERS:
# TODO: make as script argument
main_dir = os.path.join("/scratch/bailoni", "datasets/coral_data")
original_data_dir = os.path.join(main_dir, "original_data")

prediction_dir = "/scratch/bailoni/projects/coralsegm/predictions"
# ----------------

# TODO: Load from config file
datasets = {
# "HILO_part1": {'root-raw': "UH Hilo -- John Burns",
#             "dataset_name": "HILO",
#              "raw_data_type": "_plot.jpg",
#              'root-labels': "recolored_annotations/BW/UH_HILO",
#              "labels_type": "_annotation.png",
#              "images_info":
#                  "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv",
#              "experiment_name": "infer_HILO_v2_part1",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":27",
#              "val": "27:30",
#              "test": "30:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
#     "NOAA_part1": {'root-raw': "NOAA -- Couch-Oliver/cropped_images",
#         "dataset_name": "NOAA",
#          "raw_data_type": "_orthoprojection.png",
#          'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
#          "labels_type": "_annotation.png",
#          "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv",                             "experiment_name": "infer_NOAA_v1_part1",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":19",
#              "val": "19:22",
#              "test": "22:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
#     "HILO_best": {'root-raw': "UH Hilo -- John Burns",
#              "dataset_name": "HILO",
#              "raw_data_type": "_plot.jpg",
#              'root-labels': "recolored_annotations/BW/UH_HILO",
#              "labels_type": "_annotation.png",
#              "images_info":
#                  "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv",
#              "experiment_name": "infer_HILO_v2_best",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":27",
#              "val": "27:30",
#              "test": "30:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
#     "NOAA_best": {'root-raw': "NOAA -- Couch-Oliver/cropped_images",
#         "dataset_name": "NOAA",
#          "raw_data_type": "_orthoprojection.png",
#          'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
#          "labels_type": "_annotation.png",
#          "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv",                             "experiment_name": "infer_NOAA_v1_best",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":19",
#              "val": "19:22",
#              "test": "22:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
    "HILO_combined_best": {'root-raw': "UH Hilo -- John Burns",
                "dataset_name": "HILO",
             "raw_data_type": "_plot.jpg",
             'root-labels': "recolored_annotations/BW/UH_HILO",
             "labels_type": "_annotation.png",
             "images_info":
                 "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv",
             "experiment_name": "infer_HILO_NOAA_v1_best",
             # "pred_slice": "27:",
             "downscaling_factor": [1,1,1],
             "train": ":27",
             "val": "27:30",
             "test": "30:",
             "has_ignore_label": True,
             "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_combined_without_NASA")
             },
    "NOAA_combined_best": {'root-raw': "NOAA -- Couch-Oliver/cropped_images",
        "dataset_name": "NOAA",
         "raw_data_type": "_orthoprojection.png",
         'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
         "labels_type": "_annotation.png",
         "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv",                             "experiment_name": "infer_HILO_NOAA_v1_best",
             # "pred_slice": "27:",
             "downscaling_factor": [1,1,1],
             "train": ":19",
             "val": "19:22",
             "test": "22:",
             "has_ignore_label": True,
             "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_combined_without_NASA")
             },
    "HILO_combined_part1": {'root-raw': "UH Hilo -- John Burns",
                           "dataset_name": "HILO",
                           "raw_data_type": "_plot.jpg",
                           'root-labels': "recolored_annotations/BW/UH_HILO",
                           "labels_type": "_annotation.png",
                           "images_info":
                               "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv",
                           "experiment_name": "infer_HILO_NOAA_v1_part1",
                           # "pred_slice": "27:",
                           "downscaling_factor": [1, 1, 1],
                           "train": ":27",
                           "val": "27:30",
                           "test": "30:",
                           "has_ignore_label": True,
                           "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_combined_without_NASA")
                           },
    "NOAA_combined_part1": {'root-raw': "NOAA -- Couch-Oliver/cropped_images",
                           "dataset_name": "NOAA",
                           "raw_data_type": "_orthoprojection.png",
                           'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
                           "labels_type": "_annotation.png",
                           "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv",
                           "experiment_name": "infer_HILO_NOAA_v1_part1",
                           # "pred_slice": "27:",
                           "downscaling_factor": [1, 1, 1],
                           "train": ":19",
                           "val": "19:22",
                           "test": "22:",
                           "has_ignore_label": True,
                           "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_combined_without_NASA")
                           },

}


# -----------------------------
# Combined training:
# -----------------------------

# datasets = {
#     "HILO": {'root-raw': "UH Hilo -- John Burns",
#                 "dataset_name": "HILO",
#              "raw_data_type": "_plot.jpg",
#              'root-labels': "recolored_annotations/BW/UH_HILO",
#              "labels_type": "_annotation.png",
#              "images_info":
#                  "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv",
#              "experiment_name": "infer_HILO_NOAA_v1_best",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":27",
#              "val": "27:30",
#              "test": "30:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
#     "NOAA": {'root-raw': "NOAA -- Couch-Oliver/cropped_images",
#          "raw_data_type": "_orthoprojection.png",
#          'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
#          "labels_type": "_annotation.png",
#          "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv",                             "experiment_name": "infer_HILO_NOAA_v1_best",
#              # "pred_slice": "27:",
#              "downscaling_factor": [1,1,1],
#              "train": ":19",
#              "val": "19:22",
#              "test": "22:",
#              "has_ignore_label": True,
#              "input_data_dir": os.path.join(main_dir, "converted_to_hdf5_fixed_ignore_label")
#              },
# }


for dict_key in datasets:

    combined_raw = []
    combined_annotations = []
    data_info = datasets[dict_key]
    data_name = data_info["dataset_name"]
    print(data_name)
    print(data_info["experiment_name"], "...")

    input_data_dir = data_info["input_data_dir"]

    # Load GT labels:
    input_data_path = os.path.join(input_data_dir, "{}.h5".format(data_name))
    GT = var_utils.readHDF5(input_data_path, "labels")

    # Load predictions and GT labels:
    pred_dir = os.path.join(prediction_dir, data_info["experiment_name"])
    pred_path = os.path.join(pred_dir, "predictions_{}.h5".format(data_name))
    pred_segm = var_utils.readHDF5(pred_path, "data")

    # # Combine two predicted segmentations:
    # pred_segm2 = var_utils.readHDF5(pred_path.replace(".h5", "_BAK.h5"), "data")
    # pred_total = np.concatenate([pred_segm, pred_segm2], axis=0)
    # var_utils.writeHDF5(pred_total, pred_path, "data")

    # Load label-colors and original image info:
    labels_colors = pd.read_csv(input_data_path.replace(".h5", "_labels.csv"))
    images_info = pd.read_csv(input_data_path.replace(".h5", "_images_info.csv"))

    # Train-val-test slip:
    train_val_split = images_info["split"]
    image_names = images_info.iloc[:, 1]

    downscaling_crop = tuple(slice(None, None, dws_fc) for dws_fc in data_info["downscaling_factor"])
    GT = GT[downscaling_crop]
    # pred_segm = pred_segm[:6]
    pred_slice = None
    if "pred_slice" in data_info:
        pred_slice = parse_data_slice(data_info["pred_slice"])
        GT = GT[pred_slice]

    # Mask ignore-label:
    ignore_label = None
    if data_info["has_ignore_label"]:
        ignore_label = GT.max()
        ignore_mask = GT == ignore_label
        pred_segm[ignore_mask] = ignore_label

    # Convert to one-hot:
    one_hot_GT = segmutils.convert_to_one_hot(GT)
    one_hot_pred = segmutils.convert_to_one_hot(pred_segm)

    # Map labels back to original labels:
    selected_rows = labels_colors[~labels_colors['contiguous'].isnull()]
    # Mapping: {old: new}
    mapping = {int(row['contiguous']): int(row['BW']) for _, row in selected_rows.iterrows()}
    mapping[0] = 0
    remapped_segm = vigra.analysis.applyMapping(pred_segm, mapping)

    # Pad one-hot arrays to have same number of classes:
    nb_GT_classes = one_hot_GT.shape[0]
    nb_pred_classes = one_hot_pred.shape[0]
    nb_classes = max(nb_GT_classes, nb_pred_classes)
    if nb_GT_classes < nb_pred_classes:
        padding = [[0, nb_classes - nb_GT_classes]] + [[0, 0]] * len(one_hot_GT.shape[1:])
        one_hot_GT = np.pad(one_hot_GT, padding, mode="constant")
    elif nb_GT_classes > nb_pred_classes:
        padding = [[0, nb_classes - nb_pred_classes]] + [[0, 0]] * len(one_hot_pred.shape[1:])
        one_hot_pred = np.pad(one_hot_pred, padding, mode="constant")

    for crop_name in ["train", "val", "test"]:
        print(crop_name)
        crop = data_info[crop_name]
        crop_slc = var_utils.parse_data_slice(crop)

        # if pred_slice is not None:
        #     for idx, pred_cr in enumerate(pred_slice):
        #         assert isinstance(pred_cr, slice)
        #         if len(crop_slc) >= idx+1:
        #             subcrop = crop_slc[idx]
        #             print()


        onehot_crop_slc = (slice(None),) + crop_slc

        out_df = labels_colors.copy()
        # Add background row at the beginning:
        # out_df.append({"label": "background", "BW": 0}, ignore_index=True)
        out_df.loc[-1] = {"label": "background", "BW": 0}
        out_df.index = out_df.index + 1
        out_df.sort_index(inplace=True)

        # Add extra columns for IoU and confusion matrix:
        out_df = out_df.drop(columns=["Unnamed: 0", "contiguous", "R", "G", "B"])
        out_df.reindex(columns=["label", "BW", "IoU"] + list(out_df.loc[:, "label"]))

        # Compute IoU:
        IoU_global, IoU_per_class = segmutils.compute_IoU_numpy(one_hot_pred[onehot_crop_slc], one_hot_GT[onehot_crop_slc])

        # Compute and save confusion matrix:
        # By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
        #     is equal to the number of observations known to be in group :math:`i` and
        #     predicted to be in group :math:`j`. (where i is the row and j is the column)

        conf_matrix = confusion_matrix(GT[crop_slc].flatten(), pred_segm[crop_slc].flatten(), labels=list(range(nb_classes)),
                                       normalize='true')
        classes_to_be_ignored = []

        for cl, row in enumerate(conf_matrix):
            # print(cl, row.sum())
            if row.sum() <= 0.2 or cl == ignore_label:
                classes_to_be_ignored.append(cl)
                continue
            # TODO: more efficient way?
            for cl2, value in enumerate(row):
                out_df.loc[mapping[cl], out_df.loc[mapping[cl2],"label"]] = value

        # Save IoU in csv file:
        for cl, IoU_cl in enumerate(IoU_per_class):
            if cl not in classes_to_be_ignored:
                out_df.loc[mapping[cl], "IoU"] = IoU_cl


        # Save resulting scores:
        scores_dir = os.path.join(pred_dir, "scores_{}".format(data_name))
        check_dir_and_create(scores_dir)
        out_df.to_csv(os.path.join(scores_dir, "{}_scores.csv".format(crop_name)), index=False)

    # Rescale back to original res:
    if any([dws_fct != 1 for dws_fct in data_info["downscaling_factor"]]):
        remapped_segm = scipy.ndimage.zoom(remapped_segm, zoom=data_info["downscaling_factor"], order=0)

    out_segm_dir = os.path.join(pred_dir, "segm_{}".format(data_name))
    check_dir_and_create(out_segm_dir)
    for i, img_data in images_info.iterrows():
        out_path = os.path.join(out_segm_dir, "{}_{}_segm.png".format(img_data["split"], img_data[1]))
        cv2.imwrite(out_path, remapped_segm[i])

    #
    # np.set_printoptions(precision=2)
    # np.set_printoptions(suppress=True)
    # print(conf_matrix * 100)


