# coral-data-baseline
### Examples of commands to run
- Training:
- Inference: `ipython experiments/baseline/infer.py -- infer_NASA --inherit train_NASA_v1.yml --config.name_experiment infer_NASA_part3 --config.model.model_kwargs.loadfrom /scratch/bailoni/pyCh_repos/coral-data-baseline/experiments/baseline/runs/NASA_v1_part3/checkpoint.pytorch  --config.loaders.infer.loader_config.batch_size 1`
- Post-process: `ipython experiments/baseline/extra-scripts/process_predictions.py`

Another example:
```bash
ipython experiments/baseline/train_segmentation.py -- HILO_v2_tversky --inherit train_HILO_v2.yml --update0 tversky_loss.yml 
ipython experiments/baseline/train_segmentation.py -- HILO_v2_tversky_part2 --inherit train_HILO_v2.yml --update0 tversky_loss.yml --update1 retrain.yml --config.model.model_kwargs.loadfrom /scratch/bailoni/pyCh_repos/coral-data-baseline/experiments/baseline/runs/HILO_v2_tversky/checkpoint.pytorch
```
