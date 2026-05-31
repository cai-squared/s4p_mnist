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
