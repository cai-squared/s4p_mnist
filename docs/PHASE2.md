# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
This phase focuses on building, training, and validating machine learning models.

## 1. Containerization

### 1.1 Dockerfile

The Dockerfile is located in `/dockerfiles`. It is a multi-staged build that uses `python:3.11-slim-bookworm` as its base image and defines the data -> train -> predict pipeline as its entrypoint.

To build the docker image, use the `make docker_build` command. **Note: ensure that the `/data` folder is empty (does not contain any actual data) before you run the command.**

To run the docker image, ensure that are running from a directory that contains the required volumes. It should have this kind of structure:

```
(wd)
`-- data
    `-- raw
        |-- t10k-images.idx3-ubyte
        |-- t10k-labels.idx1-ubyte
        |-- train-images.idx3-ubyte
        |-- train_labels.idx3-ubyte
    `-- processed
`-- models
```

This allows the Docker container to read the data in from `\data` on your computer and write the model as an artifact to `\models`.

If you are just running from the s4p_mnist directory, you can directly use the command `make docker_run`. However, if you are running from your own local directory, then you should use the full command inside the terminal:

```
docker run -it --rm \
    -v "$(PWD)/data:/app/data" \
    -v "$(PWD)/models:/app/models" \
    s4p_mnist:latest
```

### 1.2 Environment Consistency

The docker image contains all of the requirements listed in `requirements.txt`.

## 2. Monitoring & Debugging

W&B, PDB, IPDB - Riffa

### 2.1 Monitoring
### 2.2 Debugging Practices

## 3. Profiling & Optimization

cProfile, PyTorch Profiler - Cindy

### 3.1 Python-level Profiling
### 3.2 Framework Profiling

## 4. Experiment Management & Tracking

W&B - Riffa

### 4.1 Experiment Tracking Tool

## 5. Application & Experiment Logging

logging - Saumyaa

### 5.1 Logging Setup
### 5.2 Rich Output

## 6. Configuration Management

**Owner:** Subodh (Hydra / Part F)

Phase 1 was mostly about getting the CNN to actually learn; I already had the model side working and the scores we reported in the table above. For Phase 2 the professor wanted Hydra, so this section is me wiring that requirement onto the code we already had instead of rewriting the network from scratch.

Hydra is on both entry points now, `train_model.py` and `predict_model.py`, with `@hydra.main` pointing at the same `configs/config.yaml` file the starter repo came with. Training knobs (epochs, batch size, lr, weight decay, dropout, seed, val split) plus the `paths` entries for processed MNIST and the `models/` folder all sit in the yaml. I tucked prediction under a `predict:` block in that same file so `predict_model` does not need its own mystery config.

I stayed with one yaml on purpose. The rubric asked for a second setup you can compare; for me that is just a shell override when I want a shorter run (`training.epochs=3` style) instead of duplicating almost the same file in git.

Before either script does real work it sanity-checks the merged config (epochs at least 1, dropout between 0 and 1, val split between 0 and 1, paths not blank). The checks live next to the CLIs so there is not another module to drift out of sync.

Commands that work on my machine:

```bash
make train
python -m s4p_mnist.train_model training.epochs=3 training.batch_size=256
python -m s4p_mnist.predict_model predict.output_file=reports/val_preds.csv
```

`hydra.job.chdir` is false and paths still get resolved from `PROJECT_ROOT`, so I am not fighting Hydra’s output folders when MNIST tries to load.

For the write-up: the numbers in `configs/config.yaml` are the same ones that gave us the high nineties accuracy in Phase 1. If I intentionally train for three epochs to save time, the accuracy drop is on me, not a bug.

## 7. Documentation & Repository Updates

Saumyaa

### 7.1 Updated README
### 7.2 PHASE2.md
