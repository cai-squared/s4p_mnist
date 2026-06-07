
# Multi-stage build for s4p_mnist

# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install uv
RUN uv pip install -r requirements.txt --system

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install make
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local /usr/local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV S4P_SERVE=0
ENV PORT=8080
ENV S4P_MODEL_PATH=/app/models/model.joblib

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["sh", "-c", "if [ \"$S4P_SERVE\" = \"1\" ]; then exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}; else exec sh -u flow.sh; fi"]
