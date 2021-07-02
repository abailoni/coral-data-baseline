import numpy as np
import h5py
from pathutils import get_scratch_dir
import os
import matplotlib.pyplot as plt
from PIL import Image
from segmfriends.utils.various import writeHDF5
import vigra

main_dir = os.path.join(get_scratch_dir(), "datasets/coral_data/baseline_data")



datasets = {
    "HILO": {"data": "_plot.jpg", "annotations": "_annotation.png"},
    "NASA": {"data": ".png", "annotations": "_annotation.png"},
}



for data_name in datasets:
    combined_raw = []
    combined_annotations = []
    # Load images:
    data_dir = os.path.join(main_dir, data_name, "data")
    annotations_dir = os.path.join(main_dir, data_name, "annotations")
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for file in filenames:
            if file.endswith(datasets[data_name]["data"]):
                # Load the image:
                image = np.asarray(Image.open(os.path.join(data_dir, file)))
                # Load the associated annotations:
                annotation_filename = file.replace(datasets[data_name]["data"], datasets[data_name]["annotations"])
                annotations = np.asarray(Image.open(os.path.join(annotations_dir, annotation_filename)))
                if len(combined_raw) > 0:
                    if combined_raw[-1].shape != image.shape:
                        image = np.rot90(image, axes=(0,1))
                        annotations = np.rot90(annotations, axes=(0,1))

                        # TODO: adapt method to support images of different size (and avoid this zero-padding)
                        if combined_raw[-1].shape != image.shape:
                            shape_diff = [shp_old - shp_new for shp_old, shp_new in  zip(combined_raw[-1].shape[:2], image.shape[:2])]
                            assert all([shp>=0 for shp in shape_diff])
                            image = np.pad(image, pad_width=((0,shape_diff[0]), (0,shape_diff[1]), (0,0)))
                            annotations = np.pad(annotations, pad_width=((0, shape_diff[0]), (0, shape_diff[1])))

                combined_raw.append(image)
                combined_annotations.append(annotations)

    # Reshape to pytorch-tensor style:
    combined_raw = np.stack(combined_raw)
    combined_raw = np.rollaxis(combined_raw, axis=3, start=1)
    writeHDF5(combined_raw,os.path.join(main_dir, data_name, "raw_data.h5"), "data")

    # Write outputs:
    combined_annotations = np.stack(combined_annotations)
    # Set background to zero:
    combined_annotations[combined_annotations == 164] = 0
    # FIXME: this will not preserve label-consistency across datasets!
    combined_annotations, max_label, _ = vigra.analysis.relabelConsecutive(combined_annotations)
    print("Max label for dataset {}: {}".format(data_name, max_label))
    writeHDF5(combined_annotations,os.path.join(main_dir, data_name, "annotations.h5"), "data")








#


