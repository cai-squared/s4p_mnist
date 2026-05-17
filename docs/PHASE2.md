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
    -e WANDB_API_KEY=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 \
    -v "$(PWD)/data:/app/data" \
    -v "$(PWD)/models:/app/models" \
    s4p_mnist:latest
```

### 1.2 Environment Consistency

The docker image contains all of the requirements listed in `requirements.txt`.

## 2. Monitoring & Debugging

### 2.1 Monitoring

Monitoring is done with WandB. System metrics are automatically generated.

### 2.2 Debugging Practices

## 3. Profiling & Optimization

cProfile, PyTorch Profiler - Cindy

### 3.1 Python-level Profiling
### 3.2 Framework Profiling

## 4. Experiment Management & Tracking

### 4.1 Experiment Tracking Tool

Weights & Biases is integrated into the model training. It is initialized with `wandb.init()` setting `entity="rriffaha-"` and `project="s4p-mnist"`. It logs all hyperparameters (epochs, batch_size, lr, dropout, weight_decay, val_fraction, seed) automatically via the configuration dictionary.

Each member of the team can run experiments on their own computers by running `wandb login` on their command line. If someone is running via docker, they can use Cindy's API key, which is listed in the docker section of this document.

For each experiment, WandB saves the configuration, the accuracy of the model, and the model as an artifact. Comparing runs is easy with the WandB dashboard. The final test accuracy is logged as both a metric and pinned to the run summary. The trained model is saved as a versioned W&B Artifact with hyperparameters in the metadata.

WandB is automatically enabled (in the Hydra configuration file, `training.wandb=true`). To disable WandB, one can run:

```
python -m s4p_mnist.train_model training.wandb=false
```

TODO: report link!

## 5. Application & Experiment Logging

logging - Saumyaa

### 5.1 Logging Setup
### 5.2 Rich Output

## 6. Configuration Management

Hydra is now on both entry points, `train_model.py` and `predict_model.py`, with `@hydra.main` pointing at the same `configs/config.yaml` file the starter repo came with. Training parameters (epochs, batch size, lr, weight decay, dropout, seed, val split) plus the `paths` entries for processed MNIST and the `models/` folder all sit in the yaml. Prediction is configured under a `predict:` block in that same file so `predict_model` does not need its own configuration file.

Before either script does real work it sanity-checks the configuration (epochs at least 1, dropout between 0 and 1, val split between 0 and 1, paths not blank). `hydra.job.chdir` is false and paths still get resolved from `PROJECT_ROOT`, so Hydra's output folders get created in the project directory.

The default configuration is in `configs/config.yaml` and the default hyperparameters are the same as from part 1, just wired with Hydra. To experiment, one can override the hyperparameters from the command line. For example:

```
python -m s4p_mnist.train_model training.epochs=3 training.batch_size=256
```

trains the model with the number of epochs changed to 3 and the batch size changed to 256. One can track the results of the different experiments with WandB. The default configuration gives us an accuracy of 99.5% while the customized configuration gives us an accuracy of 99.2%.

## 7. Documentation & Repository Updates

Saumyaa

### 7.1 Updated README
### 7.2 PHASE2.md
