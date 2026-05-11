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

Hydra - Subodh

### 6.1 Hydra

## 7. Documentation & Repository Updates

Saumyaa

### 7.1 Updated README
### 7.2 PHASE2.md
