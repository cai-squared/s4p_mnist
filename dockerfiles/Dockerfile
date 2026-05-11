
# Multi-stage build for s4p_mnist

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install uv
RUN uv pip install -r requirements.txt --system

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install make
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local /usr/local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["sh", "-u", "flow.sh"]
