# GuST-Diff

Official implementation for the manuscript submitted to **Advanced Engineering Informatics**:

**GuST-Diff: Guidance-Conditioned Spatio-Temporal Diffusion for Traffic Data Imputation**

GuST-Diff is a guidance-conditioned spatio-temporal diffusion model for imputing missing traffic sensor observations. The code supports point, block, and sparse missingness protocols on common traffic benchmark datasets, and provides training, checkpoint recovery, and evaluation utilities.

## Overview

Traffic data often contain random sensor dropouts, continuous block failures, and highly sparse observations. GuST-Diff addresses these settings with:

- a conditional diffusion model for probabilistic traffic data imputation;
- spatio-temporal feature conditioning with time-of-day, day-of-week, and sensor embeddings;
- optional guidance from simple interpolation signals;
- random, block, and sparse missingness generation protocols;
- evaluation with RMSE, MAE, MSE, MAPE, sMAPE, R2, and CRPS.

## Repository Structure

```text
.
|-- train.py                         # Main training and testing entry point
|-- configs/                         # YAML experiment configurations
|-- lib/
|   |-- data_processing.py           # Data loading, mask generation, and dataloaders
|   |-- enginer.py                   # Training and evaluation loops
|-- models/
|   |-- model.py                     # GuST-Diff model wrapper and diffusion process
|   |-- modules/
|       |-- guided_diffusion_unet.py  # Guided diffusion backbone
|       |-- st_diffusion_blocks.py    # Spatio-temporal diffusion blocks
|-- utils/
    |-- MetrLA_data.py               # METR-LA preprocessing
    |-- PemsBay_data.py              # PEMS-BAY preprocessing
    |-- PeMS0X_data.py               # PeMS03/04/07/08 preprocessing
    |-- imputation_pipeline.py       # Missing mask construction
```

## Requirements

The code is implemented in Python with PyTorch. A typical environment is:

```bash
conda create -n gustdiff python=3.9
conda activate gustdiff
pip install torch numpy pandas pyyaml tqdm scipy tables
```

Install the CUDA-enabled PyTorch build that matches your system from the official PyTorch instructions if GPU training is required.

## Datasets

The code expects HDF5 traffic matrices whose rows are timestamps and whose columns are sensors. The default dataset layout is:

```text
datasets/
|-- metr_la/
|   |-- metr_la.h5
|-- pems_bay/
|   |-- pems_bay.h5
|-- PeMS03/
|   |-- PeMS03.h5
|-- PeMS04/
|   |-- PeMS04.h5
|-- PeMS07/
|   |-- PeMS07.h5
|-- PeMS08/
    |-- PeMS08.h5
```

For METR-LA and PEMS-BAY, the data root can also be overridden with:

```bash
export IMPUTEST_DATA_ROOT=/path/to/metr_la        # contains metr_la.h5
export IMPUTEST_DATA_ROOT=/path/to/pems_bay       # contains pems_bay.h5
```

For PeMS03/04/07/08, set `IMPUTEST_DATA_ROOT` to the directory that contains the `PeMS03`, `PeMS04`, `PeMS07`, and `PeMS08` folders.

The public benchmark datasets are not included in this repository. Please obtain them from their official or commonly used benchmark sources and convert them to the HDF5 layout above.

## Configuration

Experiments are controlled by YAML files in `configs/`. Important fields include:

- `dataset_name`: one of `MetrLA`, `PemsBay`, `PeMS03`, `PeMS04`, `PeMS07`, `PeMS08`.
- `missing_pattern`: `point`, `block`, or `sparse`.
- `target_strategy`: training-time masking strategy, such as `random`, `block`, or `hybrid`.
- `mode`: `train` or `test`.
- `modelfolder`: an existing folder under `save/` for resuming or testing. Use an empty string for training from scratch.
- `device_id`: GPU id passed through `CUDA_VISIBLE_DEVICES`.
- `nodes`: number of sensors in the selected dataset.
- `eval_length`: window length. The provided configs use 24 time steps.
- `nsample`: number of diffusion samples used during evaluation.

Before starting a new run, check that `nodes`, `batch_size`, `device_id`, and `mode` match the intended dataset and hardware.

## Training

Set `mode: "train"` and `modelfolder: ""` in the target config, then run:

```bash
python train.py --config configs/pems04_block_conf.yaml
```

A new experiment directory will be created under:

```text
save/<dataset_name>_<missing_pattern>_<timestamp>/
```

The directory stores:

- `config_used.yaml`: backup of the configuration used for the run;
- `checkpoint_latest.pth`: latest full checkpoint for resuming training;
- `best_model.pth`: best validation model;
- `model.pth`: final model weights;
- `train_model.log`: training log.

## Resume Training

To resume an interrupted run, set:

```yaml
mode: "train"
modelfolder: "<existing_run_folder>"
```

where `<existing_run_folder>` is the folder name under `save/`, for example:

```yaml
modelfolder: "PeMS04_block_20260101_120000"
```

Then run the same command:

```bash
python train.py --config configs/pems04_block_conf.yaml
```

The script loads `save/<modelfolder>/checkpoint_latest.pth` and continues training in the same folder.

## Evaluation

To evaluate a trained model, set:

```yaml
mode: "test"
modelfolder: "<existing_run_folder>"
```

Then run:

```bash
python train.py --config configs/pems04_point_conf.yaml
```

During testing, the code loads the first available checkpoint in this order:

1. `best_model.pth`
2. `checkpoint_latest.pth`
3. `model.pth`

Evaluation outputs are written to a fresh folder under `save/` and include:

- `generated_outputs_nsample<nsample>.pk`: generated samples, targets, masks, and scalers;
- `result_nsample<nsample>.pk`: RMSE, MAE, CRPS, MSE, MAPE, sMAPE, and R2.

The metrics are also printed to the terminal.

## Missingness Protocols

The implemented missingness settings are selected by `missing_pattern`:

- `point`: random point-wise missingness;
- `block`: continuous temporal block missingness;
- `sparse`: highly sparse observations.

The `min_seq` and `max_seq` parameters control the length range of block missing segments. In the provided configs, `min_seq: 12` and `max_seq: 48` correspond to 1-4 hours for 5-minute traffic data.

## Example Commands

Train GuST-Diff on PeMS04 with block missingness:

```bash
python train.py --config configs/pems04_block_conf.yaml
```

Train GuST-Diff on METR-LA with point missingness:

```bash
python train.py --config configs/metrla_point_conf.yaml
```

Evaluate a trained PeMS08 model:

```bash
python train.py --config configs/pems08_point_conf.yaml
```

Make sure the config file has `mode: "test"` and points `modelfolder` to a valid trained run.

## Notes for Reproducibility

- The code backs up each configuration file into the corresponding run directory.
- Dataset splits follow the `train_ratio` and `val_ratio` values in the config.
- The model uses dataset-specific seeds in `lib/data_processing.py` for the implemented benchmark loaders.
- GPU nondeterminism may still occur depending on CUDA, PyTorch, and hardware settings.
- For fair comparison, use the same dataset preprocessing, missingness protocol, and `nsample` value across methods.

## Citation

If you use this code, please cite the manuscript after publication. Before publication, please refer to it as:

```bibtex
@article{gustdiff2026,
  title   = {GuST-Diff: Guidance-Conditioned Spatio-Temporal Diffusion for Traffic Data Imputation},
  journal = {Advanced Engineering Informatics},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

## Contact

For questions about the manuscript or code, please contact the authors of GuST-Diff.
