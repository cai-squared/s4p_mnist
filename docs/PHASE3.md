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

Section 3 owner: Sai Subodh Gundam Raju. We put the trained CNN online with FastAPI and Cloud Run.

### 3.1 GCP Artifact Registry

I made a Google Cloud project and turned on Artifact Registry and Cloud Run. Our API image is stored in a docker repo in `us-central1` named `s4p-mnist`. After `docker build` on my laptop I tagged and pushed the image to `us-central1-docker.pkg.dev/PROJECT_ID/s4p-mnist/api:latest`. That is the image Cloud Run pulls when the service starts.

### 3.2 Model file for serving

Training still happens with `make train` on the team machine. That writes `models/model.joblib`. The file is not in Git because it is large. For cloud deploy I copied the same file into the Docker build folder before `docker build`, or mounted `models/` when testing locally. The API reads that path from `S4P_MODEL_PATH` (default `/app/models/model.joblib` in the container).

### 3.3 FastAPI service

Code is in `api/main.py` and `api/schemas.py` as in the course template.

| Route | What it does |
|-------|----------------|
| GET `/health` | Shows if the service is up and the model loaded |
| POST `/predict` | Sends 784 pixel numbers, gets digit 0-9 |
| POST `/predict/grid` | Sends a 28 by 28 grid |
| POST `/predict/image` | Upload a small image; server resizes to 28x28 |

Local test on Windows:

```
py -3.11 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs`. A test upload of `reports/figures/test_digit_upload.png` returned digit 7 with about 99.99% confidence.

### 3.4 Cloud Run

Deploy uses the same `dockerfiles/Dockerfile` with `S4P_SERVE=1` so the container runs uvicorn instead of the training script `flow.sh`.

Example:

```
gcloud run deploy s4p-mnist-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/s4p-mnist/api:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080 \
  --set-env-vars S4P_SERVE=1
```

After deploy the team gets a public HTTPS link. We checked `/health` and `/predict/image` from the browser docs page. Cloud Run logs show each request in the GCP console.

For a quick load check we sent many requests from the Swagger page and watched latency in Cloud Run metrics.

When the class ends we delete the Cloud Run service so we are not charged.

## 4. Interactive UI

### 4.1 Streamlit or Gradio app on Hugging Face Spaces

## 5. End-to-End Demo Recording

### 5.1 Recording in main README

## 6. Documentation, Repository Updates & Cleanup

### 6.1 Comprehensive README
### 6.2 PHASE3.md
### 6.3 GCP Resource Cleanup
