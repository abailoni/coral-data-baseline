import numpy as np
import h5py
# from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5, check_dir_and_create
import vigra

# TODO: make as script argument
RELABEL_CONSECUTIVE = True

datasets = {
    "HILO": {'root-raw': "UH Hilo -- John Burns",
             "raw_data_type": "_plot.jpg",
             'root-labels': "recolored_annotations/BW/UH_HILO",
             "labels_type": "_annotation.png"},
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


main_dir = os.path.join("/scratch/bailoni", "datasets/coral_data")
original_data_dir = os.path.join(main_dir, "original_data")
out_dir = os.path.join(main_dir, "converted_to_hdf5")
check_dir_and_create(out_dir)

for data_name in datasets:
    combined_raw = []
    combined_annotations = []
    data_info = datasets[data_name]

    # Directories:
    data_dir = os.path.join(original_data_dir, data_info["root-raw"])
    labels_dir = os.path.join(original_data_dir, data_info["root-labels"])

    # Collect sizes and names of all images in folder:
    images_collected = []
    max_shape = None
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for file in filenames:
            if file.endswith(data_info["raw_data_type"]):
                if data_name == "sandin" and file.endswith("_mask.jpg"):
                    # In this data raw images do not have a unique identifier, so we need to
                    # manually ignore masks:
                    continue

                # Check shape of image:
                image = np.asarray(Image.open(os.path.join(dirpath, file)))
                # Load the associated annotations:
                annotation_filename = file.replace(data_info["raw_data_type"], data_info["labels_type"])
                annotation_path = os.path.join(labels_dir, annotation_filename)
                if not os.path.exists(annotation_path):
                    print("!!! Attention, annotation file for {} does not exist!".format(file))
                    continue

                new_image_data = {}
                new_image_data['root'] = dirpath
                new_image_data['img_path'] = os.path.join(dirpath, file)
                new_image_data['ann_path'] = annotation_path

                img = Image.open(os.path.join(dirpath, file))
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

    # TODO: for NASA dataset, set label 108 to background or ignore label
    # combined_annotations[combined_annotations == 108] = 0

    # Relabel labels consectuively to reduce size of output CNN layer
    # FIXME: problem is that then labels across datasets are no longer consistent
    if RELABEL_CONSECUTIVE:
        combined_annotations, max_label, mapping = vigra.analysis.relabelConsecutive(combined_annotations)
        print("Max label for dataset {}: {}".format(data_name, max_label))
        print(mapping)

        # Write outputs:
        writeHDF5(combined_raw, os.path.join(out_dir, "{}_consecutive.h5".format(data_name)), "image")
        writeHDF5(combined_annotations, os.path.join(out_dir, "{}_consecutive.h5".format(data_name)), "labels")
    else:
        # Write outputs:
        writeHDF5(combined_raw, os.path.join(out_dir, "{}.h5".format(data_name)), "image")
        writeHDF5(combined_annotations, os.path.join(out_dir, "{}.h5".format(data_name)), "labels")



#
