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

Section 3 owner: Sai Subodh Gundam Raju. We deployed the trained CNN with FastAPI on Google Cloud Run.

**Live service URL:** https://s4p-mnist-api-912752055925.us-central1.run.app

### 3.1 GCP Artifact Registry

I created GCP project **s4p-mnist** and enabled Artifact Registry and Cloud Run. The inference image is stored in `us-central1` under repository **s4p-mnist**. From Google Cloud Shell I built the container and pushed it to:

`us-central1-docker.pkg.dev/s4p-mnist/s4p-mnist/api:latest`

Cloud Run pulls this image when the service starts.

### 3.2 Model file for serving

Training uses `make train` on a team machine and writes `models/model.joblib`. That file is not in Git because it is large. For cloud deploy I uploaded `model.joblib` in Cloud Shell and included it in the Docker build context before `docker build`. The API reads `S4P_MODEL_PATH` (default `/app/models/model.joblib` in the container).

### 3.3 FastAPI service

Code is in `api/main.py` and `api/schemas.py`.

| Route | What it does |
|-------|----------------|
| GET `/health` | Service status and model loaded flag |
| POST `/predict` | 784 pixel values → digit 0-9 |
| POST `/predict/grid` | 28×28 grid → digit |
| POST `/predict/image` | Image upload → digit |

Local test on Windows:

```
py -3.11 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs`.

**Health check (live deploy):**

![Cloud Run health endpoint](../reports/figures/cloud_run_health.png)
*Figure 2: Live `/health` response — status ok, model loaded.*

**Predict test (live deploy):**

![Cloud Run predict via Swagger](../reports/figures/cloud_run_predict.png)
*Figure 3: POST `/predict/image` with test_digit_upload.png.*

![Cloud Run predict result](../reports/figures/cloud_run_predict_result.png)
*Figure 4: Response digit 7 with 99.99% confidence.*

### 3.4 Cloud Run

Deploy command (run in Cloud Shell):

```
gcloud run deploy s4p-mnist-api \
  --image us-central1-docker.pkg.dev/s4p-mnist/s4p-mnist/api:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080 \
  --set-env-vars S4P_SERVE=1
```

The Dockerfile uses `S4P_SERVE=1` so the container runs uvicorn instead of the training entrypoint `flow.sh`.

**Monitoring (Cloud Run metrics):**

![Cloud Run metrics dashboard](../reports/figures/cloud_run_metrics.png)
*Figure 5: Request count and latency metrics after live testing.*

When the class ends we delete the Cloud Run service to avoid charges:

```
gcloud run services delete s4p-mnist-api --region us-central1
```

## 4. Interactive UI

### 4.1 Streamlit or Gradio app on Hugging Face Spaces

## 5. End-to-End Demo Recording

### 5.1 Recording in main README

## 6. Documentation, Repository Updates & Cleanup

### 6.1 Comprehensive README
### 6.2 PHASE3.md
### 6.3 GCP Resource Cleanup
