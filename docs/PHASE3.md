# Phase 3: Continuous Machine Learning (CML) and Deployment

## Overview

## 1. Continuous Integration & Testing

### 1.1 Unit Testing with pytest
### 1.2 GitHub Actions CI Workflow
### 1.3 Pre-commit Hooks

## 2. Continuous Docker Building & CML

### 2.1 Automated Docker Builds

Docker builds are automated with this [GitHub Workflow](../.github/workflows/docker.yml). With every commit to main, the image will be built and pushed to the caisquared/s4p_mnist repository on Docker Hub.

![Screenshot of Docker Hub repository](../reports/figures/docker_hub.png)
*Figure 1: Docker Hub repository caisquared/s4p_mnist with the latest image.*

To pull the docker image, run

```
docker pull caisquared/s4p_mnist:latest
 ```

if you're running on a mac you can use the `--platform linux/amd64` flag.

To run the docker image, run

```
docker run -it --rm \
    -e WANDB_API_KEY=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 \
    -v "$(PWD)/data:/app/data" \
    -v "$(PWD)/models:/app/models" \
    caisquared/s4p_mnist:latest
```

### 2.2 Continuous Machine Learning (CML)

## 3. Deployment on Google Cloud Platform (GCP)

### 3.1 GCP Artifact Registry
### 3.2 Custom Training Job on GCP
### 3.3 FastAPI + GCP Cloud Functions
### 3.4 Dockerize & Deploy with GCP Cloud Run

## 4. Interactive UI

### 4.1 Streamlit or Gradio app on Hugging Face Spaces

## 5. End-to-End Demo Recording

### 5.1 Recording in main README

## 6. Documentation, Repository Updates & Cleanup

### 6.1 Comprehensive README
### 6.2 PHASE3.md
### 6.3 GCP Resource Cleanup
