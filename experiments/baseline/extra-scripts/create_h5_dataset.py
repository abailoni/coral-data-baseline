import numpy as np
import h5py
# from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5, check_dir_and_create, writeHDF5attribute
import vigra
import pandas as pd

# TODO: make as script argument
RELABEL_CONSECUTIVE = True

datasets = {
    "HILO": {'root-raw': "UH Hilo -- John Burns",
             "raw_data_type": "_plot.jpg",
             'root-labels': "recolored_annotations/BW/UH_HILO",
             "labels_type": "_annotation.png",
             "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv"},
    # "sandin": {'root-raw': "Sandin_SIO",
    #          "raw_data_type": ".jpg",
    #          'root-labels': "recolored_annotations/BW/Sandin-SIO",
    #          "labels_type": "_annotation.jpg"},
    # "NOAA": {'root-raw': "???", # FIXME: where is it...?
    #          "raw_data_type": "???",
    #          'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver",
    #          "labels_type": "_annotation.png"},
    # "NASA": {'root-raw': "NASA Ames NeMO Net - Alan Li/2D Projections/RGB Images",
    #          "raw_data_type": ".png",
    #          'root-labels': "recolored_annotations/BW/NASA-AlanLi",
    #          "labels_type": "_annotation.png"},

}

labels_colors = pd.read_csv("/scratch/bailoni/pyCh_repos/coral-data-baseline/data/labels_and_colors.csv")


main_dir = os.path.join("/scratch/bailoni", "datasets/coral_data")
original_data_dir = os.path.join(main_dir, "original_data")
out_dir = os.path.join(main_dir, "converted_to_hdf5")
check_dir_and_create(out_dir)

for data_name in datasets:
    combined_raw = []
    combined_annotations = []
    data_info = datasets[data_name]

    images_info = pd.read_csv(data_info["images_info"])
    images_info = images_info.sort_values(by=['split'], ascending=False, na_position='first')

    images_info.loc[images_info["split"].isna(), "split"] = "train"
    split_counts = images_info["split"].value_counts()

    train_val_split = images_info["split"]
    image_names = images_info.iloc[:, 0]

    # Directories:
    data_dir = os.path.join(original_data_dir, data_info["root-raw"])
    labels_dir = os.path.join(original_data_dir, data_info["root-labels"])

    # Collect sizes and names of all images in folder:
    images_collected = []
    max_shape = None
    for file, split in zip(image_names, train_val_split):
        if data_name == "sandin":
            raise NotImplementedError
            # if data_name == "sandin" and file.endswith("_mask.jpg"):
            #     # In this data raw images do not have a unique identifier, so we need to
            #     # manually ignore masks:
            #     continue

        # add extension and filename ending:
        file = file + data_info["raw_data_type"]


        # Check shape of image:
        image = np.asarray(Image.open(os.path.join(data_dir, file)))
        # Load the associated annotations:
        annotation_filename = file.replace(data_info["raw_data_type"], data_info["labels_type"])
        annotation_path = os.path.join(labels_dir, annotation_filename)
        if not os.path.exists(annotation_path):
            print("!!! Attention, annotation file for {} does not exist!".format(file))
            raise NotImplementedError("Train/val/test split does not support this. All images listed in csv should exist.")
            continue

        new_image_data = {}
        new_image_data['root'] = data_dir
        new_image_data['img_path'] = os.path.join(data_dir, file)
        new_image_data['ann_path'] = annotation_path

        img = Image.open(os.path.join(data_dir, file))
        img_shape = img.size[:2]

        annotations = Image.open(annotation_path)
        ann_shape = annotations.size[:2]

        assert img_shape == ann_shape, "Annotations and image do not match! {}".format(file)

        if max_shape is None:
            max_shape = img_shape
            new_image_data['rotate_image'] = False
        else:
            max_shape_diff = [0 if max_shp >= img_shape[i] else (img_shape[i] - max_shp) for i, max_shp in
                              enumerate(max_shape)]
            # Now try by rotating image:
            max_shape_diff_rot = [0 if max_shp >= img_shape[1-i] else (img_shape[1-i] - max_shp) for i, max_shp in
                             enumerate(max_shape)]
            # Check which one requires less padding:
            diff, diff_rot = np.array(max_shape_diff).sum(), np.array(max_shape_diff_rot).sum()
            new_image_data['rotate_image'] = rotate_image = diff_rot < diff

            selected_diff = max_shape_diff_rot if rotate_image else max_shape_diff
            # Now update the maximum shape:
            max_shape = [dif + max_shp  for max_shp, dif in zip(max_shape, selected_diff)]

        images_collected.append(new_image_data)

    assert len(images_collected) > 0, "No images found for dataset {}!".format(data_name)
    # Loading image in numpy inverts x and y, so invert max_shape:
    max_shape = [max_shape[1], max_shape[0]]

    # Now load the actual images into memory and pad:
    for image_data in images_collected:
        # print("Loading {}".format(image_data["img_path"]))
        image = np.asarray(Image.open(image_data['img_path']))
        annotations = np.asarray(Image.open(image_data['ann_path']))
        if image_data["rotate_image"]:
            image = np.rot90(image, axes=(0, 1))
            annotations = np.rot90(annotations, axes=(0, 1))
        if tuple(max_shape) != image.shape[:2]:
            shape_diff = [max_shp - img_shp for max_shp, img_shp in
                          zip(max_shape, image.shape[:2])]
            assert all([shp >= 0 for shp in shape_diff]), "Something went wrong with image zero padding"
            print("Zero-padded image: from {} to {}".format(image.shape[:2], max_shape))
            image = np.pad(image, pad_width=((0, shape_diff[0]), (0, shape_diff[1]), (0, 0)))
            annotations = np.pad(annotations, pad_width=((0, shape_diff[0]), (0, shape_diff[1])))
        combined_raw.append(image)
        combined_annotations.append(annotations)

    # Reshape to pytorch-tensor style:
    combined_raw = np.stack(combined_raw)
    combined_raw = np.rollaxis(combined_raw, axis=3, start=0)

    # # Normalize channels: #TODO: avoid saving in float32? Big file
    # mean = combined_raw.reshape(3, -1).mean(axis=1)[:, None, None, None]
    # std = combined_raw.reshape(3, -1).std(axis=1)[:, None, None, None]
    # combined_raw = ((combined_raw - mean) / std).astype("float32")

    # Collect some stats about annotations:
    combined_annotations = np.stack(combined_annotations)
    max_label = combined_annotations.max()
    bincount = np.bincount(combined_annotations.flatten())
    number_labels = (bincount > 0).sum()
    print("Stats for dataset {}: actual number of used labels is {}; max-label-value is {}".format(data_name, number_labels, max_label))

    if data_name == "NASA":
        # FIXME: for NASA dataset, set label 108 to background or ignore label
        raise NotImplementedError()
        # combined_annotations[combined_annotations == 108] = 0

    # Relabel labels consectuively to reduce size of output CNN layer
    # TODO: problem is that then labels across datasets are no longer consistent
    if RELABEL_CONSECUTIVE:
        labels_colors.insert(5, "contiguous", "")
        combined_annotations, max_label, mapping = vigra.analysis.relabelConsecutive(combined_annotations)
        print("Max label for dataset {}: {}".format(data_name, max_label))
        for orig_label, new_label in mapping.items():
            labels_colors.loc[labels_colors['BW'] == orig_label, "contiguous"] = new_label
    else:
        labels_colors.insert(5, "contiguous", labels_colors["BW"])

    # Write outputs:
    hdf5_path = os.path.join(out_dir, "{}.h5".format(data_name))
    writeHDF5(combined_raw, hdf5_path, "image")
    writeHDF5(combined_annotations, hdf5_path, "labels")

    # Write split info
    for split_type, count in split_counts.items():
        print(split_type, count)
        writeHDF5attribute(attribute_data=count,attriribute_name="nb_img_{}".format(split_type),
                           file_path=hdf5_path, inner_path_dataset="image")

    # Write csv files:
    labels_colors.to_csv(hdf5_path.replace(".h5", "_labels.csv"))
    images_info.to_csv(hdf5_path.replace(".h5", "_images_info.csv"))





#
