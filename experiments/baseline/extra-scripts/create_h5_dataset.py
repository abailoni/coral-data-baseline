import numpy as np
import h5py
from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5, check_dir_and_create
import vigra

main_dir = os.path.join(get_scratch_dir(), "datasets/coral_data/Coral Data Sharing")

datasets = {
    # "HILO": {'root-raw': "UH Hilo -- John Burns",
    #          "raw_data_type": "_plot.jpg",
    #          'root-labels': "recolored_annotations/BW/UH_HILO",
    #          "labels_type": "_annotation.png"},
    "NASA": {'root-raw': "NASA Ames NeMO Net - Alan Li/2D Projections/RGB Images",
             "raw_data_type": ".png",
             'root-labels': "recolored_annotations/BW/NASA-AlanLi",
             "labels_type": "_annotation.png"},
}

# datasets = {
#     "HILO": {"data": "_plot.jpg", "annotations": "_annotation.png"},
#     "NASA": {"data": ".png", "annotations": "_annotation.png"},
# }

out_dir = os.path.join(main_dir, "converted_to_hdf5")
check_dir_and_create(out_dir)

for data_name in datasets:
    combined_raw = []
    combined_annotations = []
    data_info = datasets[data_name]
    # Load images:
    data_dir = os.path.join(main_dir, data_info["root-raw"])
    labels_dir = os.path.join(main_dir, data_info["root-labels"])
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for file in filenames:
            if file.endswith(data_info["raw_data_type"]):
                print(file)
                # Load the image:
                image = np.asarray(Image.open(os.path.join(data_dir, file)))
                # Load the associated annotations:
                annotation_filename = file.replace(data_info["raw_data_type"], data_info["labels_type"])
                annotation_path = os.path.join(labels_dir, annotation_filename)
                if not os.path.exists(annotation_path):
                    print("!!! Attention, annotation file for {} does not exist!!!".format(file))
                    continue
                annotations = np.asarray(Image.open(annotation_path))
                if len(combined_raw) > 0:
                    if combined_raw[-1].shape != image.shape:
                        image = np.rot90(image, axes=(0, 1))
                        annotations = np.rot90(annotations, axes=(0, 1))

                        # TODO: adapt method to support images of different size (and avoid this zero-padding)
                        if combined_raw[-1].shape != image.shape:
                            shape_diff = [shp_old - shp_new for shp_old, shp_new in
                                          zip(combined_raw[-1].shape[:2], image.shape[:2])]
                            assert all([shp >= 0 for shp in shape_diff]), "New image is bigger than the previous (not supported yet)"
                            print("Zero-padded image: from {} to {}".format(image.shape, combined_raw[-1].shape))
                            image = np.pad(image, pad_width=((0, shape_diff[0]), (0, shape_diff[1]), (0, 0)))
                            annotations = np.pad(annotations, pad_width=((0, shape_diff[0]), (0, shape_diff[1])))

                combined_raw.append(image)
                combined_annotations.append(annotations)

    # Reshape to pytorch-tensor style:
    combined_raw = np.stack(combined_raw)
    combined_raw = np.rollaxis(combined_raw, axis=3, start=0)

    # # Normalize channels: #TODO: avoid saving in float?
    # mean = combined_raw.reshape(3, -1).mean(axis=1)[:, None, None, None]
    # std = combined_raw.reshape(3, -1).std(axis=1)[:, None, None, None]
    # combined_raw = ((combined_raw - mean) / std).astype("float32")

    # Write outputs:
    writeHDF5(combined_raw, os.path.join(out_dir, "{}.h5".format(data_name)), "image")
    combined_annotations = np.stack(combined_annotations)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FIXME: this will not preserve label-consistency across datasets!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Set background to zero:
    combined_annotations[combined_annotations == 108] = 0
    combined_annotations, max_label, _ = vigra.analysis.relabelConsecutive(combined_annotations)
    print("Max label for dataset {}: {}".format(data_name, max_label))
    writeHDF5(combined_annotations, os.path.join(out_dir, "{}.h5".format(data_name)), "labels")

#
