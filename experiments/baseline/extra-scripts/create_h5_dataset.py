import numpy as np
import h5py
# from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5, check_dir_and_create, writeHDF5attribute
import vigra
import pandas as pd
from segmfriends.io.images import write_image_to_file, write_segm_to_file
import scipy.ndimage

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# TODO: make as script argument
IGNORE_LABEL = 99
HAS_IGNORE_LABEL = True
RELABEL_CONSECUTIVE = True
RELABEL_ALL_DATASETS_CONSISTENTLY = True
NORMALIZE_RAW = True
OUT_postfix = "_combined_without_NASA"

# FIXME: At the moment I pad images and annotations with 0 (background). Better approach would be to pad them with ignore
#      label

datasets = {
    "HILO": {'root-raw': "UH Hilo -- John Burns",
             "raw_data_type": "_plot.jpg",
             'dws_ratio': 3,
             'root-labels': "recolored_annotations/BW/UH_HILO",
             "labels_type": "_annotation.png",
             "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/UH_HILO_species_stats_val_train_test_split.csv"},
    # "sandin": {'root-raw': "Sandin_SIO",
    #          "raw_data_type": ".jpg",
    #          'root-labels': "recolored_annotations/BW/Sandin-SIO", #FIXME: labels are not sharp
    #          "labels_type": "_annotation.jpg"},
    # "NASA": {'root-raw': "NASA Ames NeMO Net - Alan Li/2D Projections/RGB Images", #FIXME: some of the annotations are crap
    #          "raw_data_type": ".png",
    #          'root-labels': "recolored_annotations/BW/NASA-AlanLi",
    #          "labels_type": "_annotation.png",
    #          "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NASA_species_stats_val_train_test_split.csv"},
    "NOAA": {
        'root-raw': "NOAA -- Couch-Oliver/cropped_images",
        'dws_ratio': 3,
        # 'root-raw': "NOAA -- Couch-Oliver",
        "raw_data_type": "_orthoprojection.png",
        'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver/cropped_images",
        # 'root-labels': "recolored_annotations/BW/NOAA -- Couch-Oliver",
        "labels_type": "_annotation.png",
        "images_info": "/scratch/bailoni/pyCh_repos/coral-data-baseline/data/NOAA_species_stats_val_train_test_split.csv"},

}


def crop_empty_borders(label_image, ignore_label=99, extra_margin=1000):
    """

    :type label_image: np.ndarray
    :param ignore_label: int
    :return:
    """
    assert label_image.ndim == 2
    foreground_mask = label_image != ignore_label
    crops = []
    for axis in range(2):
        foreground_indices = np.argwhere(foreground_mask.sum(axis=1 - axis) > 0)
        left_crop = foreground_indices.min() - extra_margin
        left_crop = left_crop if left_crop >= 0 else 0
        right_crop = foreground_indices.max() + extra_margin
        right_crop = right_crop if right_crop < label_image.shape[axis] else label_image.shape[axis]
        crops.append(slice(left_crop, right_crop))
    return tuple(crops)


def main_function(mode="convert_to_hdf5"):
    assert mode in ["convert_to_hdf5", "crop_ignore_label"], "Mode is not recognized"

    labels_colors = pd.read_csv("/scratch/bailoni/pyCh_repos/coral-data-baseline/data/labels_and_colors.csv")

    original_data_dir = "/g/scb/alexandr/shared/alberto/datasets/coral_data/Coral Data Sharing"
    main_dir = os.path.join("/scratch/bailoni", "datasets/coral_data")
    out_dir = os.path.join(main_dir, "converted_to_hdf5" + OUT_postfix)
    check_dir_and_create(out_dir)

    collected_data = {}

    for data_name in datasets:
        combined_raw = []
        combined_annotations = []
        data_info = datasets[data_name]

        images_info = pd.read_csv(data_info["images_info"], skipfooter=1)
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
                if data_name == "sandin" and file.endswith("_mask.jpg"):
                    # In this data raw images do not have a unique identifier, so we need to
                    # manually ignore masks:
                    continue
                raise NotImplementedError("Labels are not sharp, please update them")

            # add extension and filename ending:
            file = file + data_info["raw_data_type"]

            # Check shape of image:
            img = Image.open(os.path.join(data_dir, file))
            # Load the associated annotations:
            annotation_filename = file.replace(data_info["raw_data_type"], data_info["labels_type"])
            annotation_path = os.path.join(labels_dir, annotation_filename)
            if not os.path.exists(annotation_path):
                print("!!! Attention, annotation file for {} does not exist!".format(file))
                raise NotImplementedError(
                    "Train/val/test split does not support this. All images listed in csv should exist.")
                continue

            new_image_data = {}
            new_image_data['root'] = data_dir
            new_image_data['img_path'] = os.path.join(data_dir, file)
            new_image_data['ann_path'] = annotation_path

            img_shape = img.size[:2]

            annotations = Image.open(annotation_path)
            ann_shape = annotations.size[:2]

            assert img_shape == ann_shape, "Annotations and image do not match! {}".format(file)

            if mode == "crop_ignore_label":
                img = np.asarray(img)
                annotations = np.asarray(annotations)
                crop_slice = crop_empty_borders(annotations, ignore_label=IGNORE_LABEL, extra_margin=1000)
                annotations = annotations[crop_slice]
                img = img[crop_slice]

                # Write images:
                out_cropped_dir = os.path.join(data_dir, "cropped_images")
                check_dir_and_create(out_cropped_dir)
                out_label_dir = os.path.join(labels_dir, "cropped_images")
                check_dir_and_create(out_label_dir)
                write_segm_to_file(os.path.join(out_label_dir, annotation_filename), annotations)
                write_image_to_file(os.path.join(out_cropped_dir, file), img)

                continue

            if max_shape is None:
                max_shape = img_shape
                new_image_data['rotate_image'] = False
            else:
                max_shape_diff = [0 if max_shp >= img_shape[i] else (img_shape[i] - max_shp) for i, max_shp in
                                  enumerate(max_shape)]
                # Now try by rotating image:
                max_shape_diff_rot = [0 if max_shp >= img_shape[1 - i] else (img_shape[1 - i] - max_shp) for i, max_shp
                                      in
                                      enumerate(max_shape)]
                # Check which one requires less padding:
                diff, diff_rot = np.array(max_shape_diff).sum(), np.array(max_shape_diff_rot).sum()
                new_image_data['rotate_image'] = rotate_image = diff_rot < diff

                selected_diff = max_shape_diff_rot if rotate_image else max_shape_diff
                # Now update the maximum shape:
                max_shape = [dif + max_shp for max_shp, dif in zip(max_shape, selected_diff)]

            images_collected.append(new_image_data)

        if mode == "crop_ignore_label":
            return

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

            # Downscale images if necessary:
            # TODO: this should be done before to improve performaces:
            if 'dws_ratio' in datasets[data_name]:
                dws_ratio = datasets[data_name]['dws_ratio']
                assert isinstance(dws_ratio, int)
                if dws_ratio != 1:
                    # Apply filter to image and downsample:
                    image = np.stack(
                        [scipy.ndimage.uniform_filter(image[..., ch], size=dws_ratio)[::dws_ratio, ::dws_ratio] for ch
                         in range(image.shape[2])], axis=2)
                    annotations = annotations[::dws_ratio, ::dws_ratio]

            combined_raw.append(image)
            combined_annotations.append(annotations)

        # Reshape to pytorch-tensor style:
        combined_raw = np.stack(combined_raw)
        combined_raw = np.rollaxis(combined_raw, axis=3, start=0)
        # FIXME: If the image was RGBA, throw away the Alpha channel
        if combined_raw.shape[0] == 4:
            combined_raw = combined_raw[:3]
        assert combined_raw.shape[0] == 3

        # Collect some stats about annotations:
        combined_annotations = np.stack(combined_annotations)
        # max_label = combined_annotations.max()
        # bincount = np.bincount(combined_annotations.flatten())
        # number_labels = (bincount > 0).sum()
        # print("Stats for dataset {}: actual number of used labels is {}; max-label-value is {}".format(data_name, number_labels, max_label))

        if data_name == "NASA":
            # Set OutofBounds (99), Bare Substratum (30), and no data (108) classes to background:
            # In this case, ignore label can be safely mapped to background because the image there looks very different
            combined_annotations[combined_annotations == 99] = 0
            combined_annotations[combined_annotations == 30] = 0
            combined_annotations[combined_annotations == 108] = 0

        print("Total size of combined dataset {}: ".format(data_name), combined_annotations.shape)
        collected_data[data_name] = [combined_raw, combined_annotations]

        # Write some image data:
        images_info.to_csv(os.path.join(out_dir, "{}_images_info.csv".format(data_name)))

        # Write split info
        crop_indx = 0
        print("Split counts for {}:".format(data_name))
        for split_type in ["train", "val", "test"]:
            count = split_counts[split_type]
            print(
                "{} - Number of images: {} - Crop slice {}:{}".format(split_type, count, crop_indx, crop_indx + count))
            crop_indx += count
            # writeHDF5attribute(attribute_data=count,attriribute_name="nb_img_{}".format(split_type),
            #                    file_path=hdf5_path, inner_path_dataset="image")

    def relabel_continuous_with_ignore_label(labels, ignore_label=IGNORE_LABEL):
        """
        Background is automatically preserved in the mapping transformation, but the ignore label requires extra care
        """
        # Temporarely map the ignore label to zero:
        ignore_mask = labels == ignore_label
        labels[ignore_mask] = 0

        # Now remap using vigra:
        remapped, max_label, mapping = vigra.analysis.relabelConsecutive(labels)

        # Finally, map ignore label as max_label+1:
        remapped[ignore_mask] = max_label + 1
        mapping[ignore_label] = max_label + 1

        return remapped, max_label + 1, mapping

    # Check which labels appear across all datasets:
    unique_labels = None
    labels_colors.insert(5, "contiguous", "")
    if len(datasets) > 1 and RELABEL_CONSECUTIVE and RELABEL_ALL_DATASETS_CONSISTENTLY:
        for data_name in datasets:
            new_labels = np.unique(collected_data[data_name][1])
            if unique_labels is None:
                unique_labels = new_labels
            else:
                unique_labels = np.concatenate([unique_labels, new_labels], axis=0)

        # Now define the mapping:
        _, max_label, mapping = relabel_continuous_with_ignore_label(unique_labels)

        has_ignore_label = IGNORE_LABEL in mapping
        print("Ignore label {} is in dataset {}: {}".format(IGNORE_LABEL, data_name, has_ignore_label))
        if has_ignore_label:
            print("   --> Ignore label {} mapped to label {}".format(IGNORE_LABEL, mapping[IGNORE_LABEL]))

        # Write mapping to csv table:
        print("Max label for all datasets: {}. Total number of out channels/classes for the model: {}".format(max_label,
                                                                                                              max_label + 1 if not has_ignore_label else max_label))
        for orig_label, new_label in mapping.items():
            labels_colors.loc[labels_colors['BW'] == orig_label, "contiguous"] = new_label

    # Now do the actual continuous relabeling and write the outputs:
    for data_name in datasets:
        combined_raw = collected_data[data_name][0]
        combined_annotations = collected_data[data_name][1]
        # Relabel labels consectuively to reduce size of output CNN layer
        if unique_labels is not None:
            # Apply the mapping that was previously computed
            vigra.analysis.applyMapping(combined_annotations, mapping, out=combined_annotations)
        elif RELABEL_CONSECUTIVE:
            combined_annotations, max_label, mapping = relabel_continuous_with_ignore_label(combined_annotations)

            has_ignore_label = IGNORE_LABEL in mapping
            print("Ignore label {} is in dataset {}: {}".format(IGNORE_LABEL, data_name, has_ignore_label))
            if has_ignore_label:
                print("   --> Ignore label {} mapped to label {}".format(IGNORE_LABEL, mapping[IGNORE_LABEL]))

            print(
                "Max label for dataset {}: {}. Total number of out channels/classes for the model: {}".format(data_name,
                                                                                                              max_label,
                                                                                                              max_label + 1 if not has_ignore_label else max_label))
            labels_colors = labels_colors.assign(contiguous="")
            for orig_label, new_label in mapping.items():
                labels_colors.loc[labels_colors['BW'] == orig_label, "contiguous"] = new_label
        else:
            labels_colors = labels_colors.assign(contiguous=labels_colors["BW"])

        # Write outputs:
        hdf5_path = os.path.join(out_dir, "{}.h5".format(data_name))
        writeHDF5(combined_raw, hdf5_path, "image")
        # Normalize channels:
        if NORMALIZE_RAW:
            mean = combined_raw.reshape(3, -1).mean(axis=1)[:, None, None, None]
            std = combined_raw.reshape(3, -1).std(axis=1)[:, None, None, None]
            combined_raw = ((combined_raw - mean) / std).astype("float16")
            writeHDF5(combined_raw, hdf5_path, "image_normalized")

        writeHDF5(combined_annotations, hdf5_path, "labels")

        # Write csv files:
        labels_colors.to_csv(hdf5_path.replace(".h5", "_labels.csv"))


# main_function("crop_ignore_label")
main_function()

#
