# Configs Directory

Store Hydra configuration files and experiment configurations here.

## Structure

Organize configs by component:
- `model/` — Model architecture and hyperparameters
- `data/` — Data loading and preprocessing configs
- `training/` — Training loops and optimization settings
- `experiment/` — Full experiment configurations

## Usage

Hydra automatically loads configs from this directory. Override values via command line:

```bash
python train.py model=bert data.batch_size=32
```

## Phase

Phase 2 deliverable — Experiment configuration framework.
