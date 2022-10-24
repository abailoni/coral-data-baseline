# Creating the input hdf5 datasets

## Generating the hdf5 datasets
Run the `create_h5_dataset.py` script [here](https://github.com/abailoni/coral-data-baseline/blob/main/experiments/baseline/extra-scripts/create_h5_dataset.py). It includes quite some spaghetti code, but all the options can be modified at the beginning of the file and they should be self-explanatory enough ;)

:::{important}
At the end, the script will print some important information that you will need to provide in the training config files (next section). 
:::

## Preparing the training config files
All the training parameters are set from a big `.yaml` config files, which can be found in the `experiments/baseline/configs` [folder](https://github.com/abailoni/coral-data-baseline/tree/main/experiments/baseline/configs).

I created different configs for each training setup and dataset:

- `train_NASA_v1.yml` for example was the one for training on NASA dataset
- `train_HILO_NOAA_v1.yml` was for training both on HILO and NOAA datasets

There are some comments in the configs, but probably not very self-explanatory. Here there is a list of things that you should update:

- **Paths to the HDF5 datasets:** obviously, update all paths that refer to the dataset you generated in the previous step
- **Model parameters**: in the config section with the model parameters, you probably need to update the following ones 
  - _loadfrom_: If you are training from scratch, comment this line. Otherwise you should provide the path to the previously trained model
  - _out_channels_: This entry specify the number of output channels of the model. Please insert the number provided by the `create_h5_dataset.py` script (this number will include background label).

### Train/val/test splits
Another important thing to set in the config files are the train/val/test splits.

The loader section of the config has three parts : traing, val, and infer.

## Where are training data saved

