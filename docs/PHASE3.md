# Phase 3: Continuous Machine Learning (CML) and Deployment

## Overview

## 1. Continuous Integration & Testing


### 1.1 Unit Testing with pytest

**File:** `tests/test_data.py`

Tests cover the entire data layer: the IDX binary parser (`src/s4p_mnist/data/loaders.py`) and the pipeline CLI (`src/s4p_mnist/data/make_dataset.py`). The 34 tests are organized into 8 classes:

| Class | What it tests |
|-------|--------------|
| `TestReadIdxImages` | IDX3 parser: shape, dtype, value roundtrip, missing file, bad magic, truncated header, body mismatch |
| `TestReadIdxLabels` | IDX1 parser: same coverage as images |
| `TestLoadRaw` | All four array shapes/dtypes, train/test size mismatch, label out of range, missing file |
| `TestSaveAndLoadProcessed` | File creation, value/dtype roundtrip, missing files error with `make data` hint, auto-creates directory |
| `TestProcessedFilesExist` | All present, empty dir, partial presence |
| `TestMakeDataset` | Creates files, idempotent (no-force skips), force triggers reprocessing, missing raw raises |
| `TestMain` | CLI exit codes 0/1, `--force` flag accepted |

All tests are self-contained: synthetic IDX binary files are generated on the fly using pytest's `tmp_path` fixture so the real MNIST data files are never required in CI.

To run locally:

```
uv run pytest tests/test_data.py -v
```

![34 tests passing](../reports/figures/pytest_34_passed.png)
*Figure: All 34 data layer tests passing locally (Python 3.11.9, pytest 9.0.3).*

---

### 1.2 GitHub Actions CI Workflow

**File:** `.github/workflows/ci.yml`

The CI workflow triggers on every push and pull request to `main` and runs the following steps against Python 3.11:

1. Install dependencies from `requirements.txt` and `requirements_dev.txt`
2. `ruff check .` — lint
3. `ruff format --check .` — format check
4. `pip install -e .` — install the project package
5. `mypy src/s4p_mnist` — static type checking
6. `pytest tests/ --cov=s4p_mnist --cov-report=xml` — run all tests with coverage
7. Upload coverage report to Codecov

The workflow ensures that every PR to `main` is lint-clean, type-safe, and fully tested before merging.

![CI green on PR #18](../reports/figures/ci_green_pr18.png)

*Figure: Both CI checks green on PR #18 — lint-and-test (3.11) and build-and-push.*

---

### 1.3 Pre-commit Hooks

**File:** `.pre-commit-config.yaml`

Pre-commit hooks enforce code quality before any commit lands. Three hook sets are active:

| Hook | What it enforces |
|------|-----------------|
| `ruff` (with `--fix`) | Auto-fixes lint errors on commit |
| `ruff-format` | Consistent code formatting |
| `mypy` | Type checking with `--ignore-missing-imports` |
| `trailing-whitespace` | No trailing spaces |
| `end-of-file-fixer` | Files end with a newline |
| `check-yaml` | Valid YAML syntax |

To install the hooks in your local clone:

```
pip install pre-commit
pre-commit install
```

After installation, every `git commit` runs the checks automatically. To run manually against all files:

```
pre-commit run --all-files
```

![Pre-commit config](../reports/figures/pre_commit_config.png)
*Figure: `.pre-commit-config.yaml` at repo root with ruff, mypy, and standard hooks.*

---

## 2. Continuous Docker Building & CML

### 2.1 Automated Docker Builds

Docker builds are automated with this [GitHub Workflow](../.github/workflows/docker.yml). With every commit to main, the image will be built and pushed to the caisquared/s4p_mnist repository on Docker Hub.

![Screenshot of Docker Hub repository](../reports/figures/docker_hub.png)
*Figure: Docker Hub repository caisquared/s4p_mnist with the latest image.*

To pull the docker image, run

```
docker pull caisquared/s4p_mnist:latest
 ```

if you're running on a mac you can use the `--platform linux/amd64` flag.

To run the docker image, run

```
docker run -it --rm \
    --env-file .env \
    -v "$(PWD)/data:/app/data" \
    -v "$(PWD)/models:/app/models" \
    caisquared/s4p_mnist:latest
```

### 2.2 Continuous Machine Learning (CML)

CML is integrated in this [GitHub Workflow](../.github/workflows/cml.yml) so that a PR triggers a model training and posts a comment back on the PR with a link to the W&B run, the classification report, and the confusion matrix.

An example PR is at this link: [PR 23](https://github.com/cai-squared/s4p_mnist/pull/23)

![Screenshot of the PR comment](../reports/figures/pr_comment.png)
*Figure: PR comment with W&B link and classification report visible.*

## 3. Deployment on Google Cloud Platform (GCP)

Section 3 owner: Sai Subodh Gundam Raju.

**Cloud Run URL (3.4):** https://s4p-mnist-api-912752055925.us-central1.run.app

### 3.1 GCP Artifact Registry

GCP project **s4p-mnist** with Artifact Registry enabled. The inference image is in `us-central1`, repository **s4p-mnist**:

`us-central1-docker.pkg.dev/s4p-mnist/s4p-mnist/api:latest`

Cloud Run and Vertex AI custom jobs pull this image.

![Artifact Registry](../reports/figures/artifact_registry.png)
*Figure: Artifact Registry repository s4p-mnist showing the api:latest image.*

### 3.2 Custom Training Job on GCP

Processed MNIST arrays live in Cloud Storage. A Vertex AI **Custom job** runs training inside the same container image, reading data from GCS and writing the trained model back to GCS.

**GCS bucket (dataset + model output):**

- `gs://s4p-mnist-training-data/data/processed/` — `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`
- `gs://s4p-mnist-training-data/models/model.joblib` — trained model written after the job finishes

Upload from Cloud Shell (after `make data` locally or in Shell):

```
gsutil mb -l us-central1 gs://s4p-mnist-training-data
gsutil -m cp data/processed/*.npy gs://s4p-mnist-training-data/data/processed/
```

**Vertex AI custom job** (Cloud Shell — paste all at once):

```
cat > /tmp/s4p-mnist-train.yaml << 'EOF'
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
  replicaCount: 1
  containerSpec:
    imageUri: us-central1-docker.pkg.dev/s4p-mnist/s4p-mnist/api:latest
    command:
    - python
    args:
    - -m
    - s4p_mnist.train_model
    - training.wandb=false
    env:
    - name: S4P_GCS_DATA_URI
      value: gs://s4p-mnist-training-data/data/processed
    - name: S4P_GCS_MODEL_OUTPUT_URI
      value: gs://s4p-mnist-training-data/models/model.joblib
EOF

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=s4p-mnist-train \
  --config=/tmp/s4p-mnist-train.yaml
```

Do **not** put `container-command` inside `--worker-pool-spec` — gcloud rejects it. Use `--config` for env vars, or `--command=python` with `--args=-m,s4p_mnist.train_model,training.wandb=false` if you skip GCS env vars.

Training code syncs `S4P_GCS_DATA_URI` into `data/processed/` before training and uploads the model to `S4P_GCS_MODEL_OUTPUT_URI` when done (`src/s4p_mnist/train_model.py`, `src/s4p_mnist/utils/io.py`).

![GCS bucket with MNIST data](../reports/figures/gcs_bucket_data.png)
*Figure: Cloud Storage bucket `s4p-mnist-training-data/data/processed/` with processed .npy files.*

![GCS model output](../reports/figures/gcs_model_output.png)
*Figure: Trained `model.joblib` saved to `s4p-mnist-training-data/models/` after the custom job.*

![Vertex AI custom job succeeded](../reports/figures/vertex_custom_job_succeeded.png)
*Figure: Completed Vertex AI custom training job.*

### 3.3 FastAPI on Cloud Functions

FastAPI code is in `api/main.py` and `api/schemas.py`. The same routes run on **Cloud Functions Gen 2** via the `handler` entry point (Mangum ASGI adapter).

| Route | What it does |
|-------|----------------|
| GET `/health` | Service status and model loaded flag |
| POST `/predict` | 784 pixel values → digit 0-9 |
| POST `/predict/grid` | 28×28 grid → digit |
| POST `/predict/image` | Image upload → digit |

Deploy from repo root in Cloud Shell (clone first if needed):

```
git clone https://github.com/cai-squared/s4p_mnist.git
cd s4p_mnist

gcloud functions deploy s4p-mnist-api-fn \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=handler \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300s \
  --set-env-vars=S4P_GCS_MODEL_URI=gs://s4p-mnist-training-data/models/model.joblib
```

Cloud Functions requires `main.py` at the repo root (re-exports `handler` from `api/main.py`). Run the command from inside the cloned repo, not from `~`.

**Cloud Functions URL:** https://us-central1-s4p-mnist.cloudfunctions.net/s4p-mnist-api-fn

Local test on Windows:

```
py -3.11 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs`.

![Cloud Functions deployed](../reports/figures/cloud_functions_deployed.png)
*Figure: Cloud Functions Gen 2 service s4p-mnist-api-fn deployed.*

![Cloud Functions health endpoint](../reports/figures/cloud_functions_health.png)
*Figure: Live Cloud Functions `/health` response — status ok, model loaded.*

### 3.4 Cloud Run

FastAPI is also deployed on **Cloud Run** for production (Hugging Face Gradio app uses this URL).

Deploy command (run in Cloud Shell):

```
gcloud run deploy s4p-mnist-api \
  --image us-central1-docker.pkg.dev/s4p-mnist/s4p-mnist/api:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080 \
  --set-env-vars S4P_SERVE=1,S4P_GCS_MODEL_URI=gs://s4p-mnist-training-data/models/model.joblib
```

The Dockerfile uses `S4P_SERVE=1` so the container runs uvicorn instead of the training entrypoint `flow.sh`.

**Health check (live deploy):**

![Cloud Run health endpoint](../reports/figures/cloud_run_health.png)
*Figure: Live Cloud Run `/health` response — status ok, model loaded.*

**Predict test (live deploy):**

![Cloud Run predict via Swagger](../reports/figures/cloud_run_predict.png)
*Figure: POST `/predict/image` with test_digit_upload.png.*

![Cloud Run predict result](../reports/figures/cloud_run_predict_result.png)
*Figure: Response digit 7 with 99.99% confidence.*

**Monitoring (Cloud Run metrics):**

![Cloud Run metrics dashboard](../reports/figures/cloud_run_metrics.png)
*Figure: Request count and latency metrics after live testing.*

When the class ends we delete Cloud Run and Cloud Functions services to avoid charges:

```
gcloud run services delete s4p-mnist-api --region us-central1
gcloud functions delete s4p-mnist-api-fn --region us-central1 --gen2
```

## 4. Interactive UI

### 4.1 Gradio app on Hugging Face Spaces

The repository now includes a live demo app at `app.py` that calls the Cloud Run deployment endpoint for predictions.

- The app sends each drawn or uploaded image to the backend inference endpoint.
- The app displays the predicted digit and confidence score in a friendly interface.

This deployment is wired into GitHub Actions so pushing to `main` updates the Space automatically.

The redeploy workflow is at `.github/workflows/hf_spaces.yml`

Try the app out for yourself [here](https://huggingface.co/spaces/caisquared/s4p_mnist).

![Hugging Face app](../reports/figures/hf_app.png)
*Figure: Live Gradio app on Hugging Face Spaces.*

## 5. End-to-End Demo Recording

### 5.1 Recording in main README

A YouTube video demonstrating the project has been added at the top of the README (under the HuggingFace metadata).

## 6. Documentation, Repository Updates & Cleanup

### 6.1 Comprehensive README

The base README has been updated with a Phase 3 section that links to this document and summarizes the new tools and services added in this phase (GitHub Actions CI/CD, automated Docker builds, CML, GCP Artifact Registry and Cloud Run, FastAPI, and the Gradio app on Hugging Face Spaces). The end-to-end demo recording is embedded near the top of the README as a **Live Demonstration** link.

![Rendered README](../reports/figures/readme_rendered.png)
*Figure: Main README rendered on GitHub, showing the Live Demonstration link near the top.*

### 6.2 PHASE3.md

This document (`docs/PHASE3.md`) records each Phase 3 deliverable with its file/directory reference, a screenshot of the working result, and a short explanation. It is linked from
the main README.

### 6.3 GCP Resource Cleanup

Section 3 owner: Sai Subodh Gundam Raju.

- [x] **Services stopped / resources removed**
- [x] **Evidence:** screenshot of empty/cleaned GCP console + explanation below

After final submission, all billable GCP resources from Section 3 were deleted: Cloud Run (`s4p-mnist-api`), Cloud Functions (`s4p-mnist-api-fn`), GCS bucket (`s4p-mnist-training-data`), and Artifact Registry repo (`s4p-mnist`, `us-central1`). The Vertex AI job `s4p-mnist-train` was already **Finished** and left in the console for grading only.

**Explanation:** Live endpoints and storage were removed so the `s4p-mnist` project would not keep charging after the course. Cloud Run and Cloud Functions were deleted first; then the training bucket and Docker image repository were removed.

**Commands (Cloud Shell):**

```
gcloud config set project s4p-mnist
gcloud run services delete s4p-mnist-api --region=us-central1 --quiet
gcloud functions delete s4p-mnist-api-fn --region=us-central1 --gen2 --quiet
gsutil -m rm -r gs://s4p-mnist-training-data
gcloud artifacts repositories delete s4p-mnist --location=us-central1 --quiet
```

![GCP cleanup](../reports/figures/gcp_cleanup.png)
*Figure: GCP console after cleanup — no Cloud Run services remaining.*
