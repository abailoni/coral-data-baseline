import sys
import os

import sys
from copy import deepcopy
import os
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

# Imports for models/criteria
import neurofire
import confnets
import inferno
import segmfriends


from inferno.io.transform import Compose
from neurofire.criteria.loss_wrapper import LossWrapper
from speedrun import BaseExperiment, AffinityInferenceMixin
from speedrun.log_anywhere import register_logger
from speedrun.py_utils import create_instance

from segmfriends.utils.config_utils import recursive_dict_update
from segmfriends.utils.various import check_dir_and_create, writeHDF5

from segmfriends.datasets.mutli_scale import MultiScaleDataset, collate_indices, MultiScaleDatasets


# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class BaseBatterySegmExperiment(BaseExperiment, AffinityInferenceMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseBatterySegmExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)


        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup(update_git_revision=False)

        # register_logger(FirelightLogger, "image")
        register_logger(self, 'scalars')

        self.model_class = self.get('model/model_class')

        self.set_devices()

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config

        assert "model_class" in model_config
        assert "model_kwargs" in model_config
        model_class = model_config["model_class"]
        model_kwargs = model_config["model_kwargs"]
        model_path = model_kwargs.pop('loadfrom', None)
        model_config = {model_class: model_kwargs}
        model = create_instance(model_config, self.MODEL_LOCATIONS)

        if model_path is not None:
            print(f"loading model from {model_path}")
            loaded_model = torch.load(model_path)["_model"]
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict)

        return model

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_kwargs = self.get("trainer/criterion/kwargs", {})
        # from vaeAffs.models.losses import EncodingLoss, PatchLoss, PatchBasedLoss, StackedAffinityLoss
        loss_name = self.get("trainer/criterion/loss_name",
                             "inferno.extensions.criteria.set_similarity_measures.SorensenDiceLoss")
        loss_config = {loss_name: loss_kwargs}

        criterion = create_instance(loss_config, self.CRITERION_LOCATIONS)
        transforms = self.get("trainer/criterion/transforms")
        if transforms is not None:
            assert isinstance(transforms, list)
            transforms_instances = []
            # Build transforms:
            for transf in transforms:
                transforms_instances.append(create_instance(transf, []))
            # Wrap criterion:
            criterion = LossWrapper(criterion, transforms=Compose(*transforms_instances))

        self._trainer.build_criterion(criterion)
        self._trainer.build_validation_criterion(criterion)

    def set_devices(self):
        # --------- In case of multiple GPUs: ------------
        n_gpus = torch.cuda.device_count()
        gpu_list = range(n_gpus)
        self.set("gpu_list", gpu_list)
        self.trainer.cuda(gpu_list)

        # # --------- Debug on trendytukan, force to use only GPU 0: ------------
        # self.set("gpu_list", [0])
        # self.trainer.cuda([0])

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        datasets = MultiScaleDatasets.from_config(kwargs)
        return DataLoader(datasets, **kwargs.get("loader_config", {}))

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        datasets = MultiScaleDatasets.from_config(kwargs)
        return DataLoader(datasets, **kwargs.get("loader_config", {}))

    def build_infer_loader(self):
        kwargs = deepcopy(self.get('loaders/infer'))
        loader_config = kwargs.get('loader_config')
        loader_config["collate_fn"] = collate_indices
        dataset = MultiScaleDataset.from_config(kwargs)
        return DataLoader(dataset, **kwargs.get("loader_config", {}))

    def save_infer_output(self, output):
        print("Saving....")
        dir_path = os.path.join("/scratch/bailoni/projects/coralsegm/predictions", self.get("name_experiment", default="generic_experiment"))
        check_dir_and_create(dir_path)
        filename = os.path.join(dir_path, "predictions_{}.h5".format(self.get("loaders/infer/dataset_name")))
        writeHDF5(output.astype(np.float16), filename, self.get("inner_path_output", 'data'))
        print("Saved to ", filename)

        # Dump configuration to export folder:
        self.dump_configuration(os.path.join(dir_path, "prediction_config_{}.yml".format(self.get("loaders/infer/dataset_name"))))


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    # # Update HCI_HOME paths:
    # for i, key in enumerate(sys.argv):
    #     if "HCI__HOME" in sys.argv[i]:
    #         sys.argv[i] = sys.argv[i].replace("HCI__HOME/", get_abailoni_hci_home_path())
    #
    # # Update RUNS paths:
    # for i, key in enumerate(sys.argv):
    #     if "RUNS__HOME" in sys.argv[i]:
    #         sys.argv[i] = sys.argv[i].replace("RUNS__HOME", experiments_path)


    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = os.path.join(config_path, sys.argv[i])
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[ind])
            i += 1
        else:
            break
    cls = BaseBatterySegmExperiment
    cls().infer()



# TODO: move somewhere else!
def compute_IoU(predictions, targets):
    """
    The shape of both `predictions` and `targets` should be (batch_size, nb_classes, x_size_image, y_size_image)
    """
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert predictions.ndim == 4 and targets.ndim == 4

    ### Start of your code ###
    # We first threshold the values, and then reshape the arrays:
    nb_classes = predictions.shape[1]
    # 0 --> 1
    # 1 --> 0
    # 2,2
    # 3,3
    predictions = (predictions > 0.5).permute(1, 0, 2, 3).reshape(nb_classes, -1)
    targets = (targets > 0.5).permute(1, 0, 2, 3).view(nb_classes, -1)  # (nb_classes, batch * x * y)

    # Intersection: both GT and predictions are True (AND operator &)
    # Union: at least one of the two is True (OR operator |)
    IoU = 0
    for cl in range(nb_classes):
        IoU = IoU + (predictions[cl] & targets[cl]).sum().float() / (predictions[cl] | targets[cl]).sum().float()
    IoU = IoU / nb_classes
    ### End of your code ###

    return IoU
