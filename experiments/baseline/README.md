# coral-data-baseline
### Examples of commands to run
- Training:
- Inference: `ipython experiments/baseline/infer.py -- infer_NASA --inherit train_NASA_v1.yml --config.name_experiment infer_NASA_part3 --config.model.model_kwargs.loadfrom /scratch/bailoni/pyCh_repos/coral-data-baseline/experiments/baseline/runs/NASA_v1_part3/checkpoint.pytorch  --config.loaders.infer.loader_config.batch_size 1`
- Post-process: `ipython experiments/baseline/extra-scripts/process_predictions.py`
