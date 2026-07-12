# RAIN-Net v0.1 Dockerfile -- single-stage, slim Python image.
# Built artifact serves the HTTP API on port 8080.
#
# Build:
#   docker build -t rain-net:0.1 .
# Run:
#   docker run --rm -p 8080:8080 rain-net:0.1
# With auth + custom skills dir:
#   docker run --rm -p 8080:8080 \
#     -e RAIN_AUTH_TOKEN=mysecret \
#     -v $(pwd)/skills:/app/skills \
#     rain-net:0.1
FROM python:3.12-slim

# Build deps (kept minimal; no torch in default image so it stays small).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed for install + runtime (not the .venv, target/,
# data/checkpoints/, etc).
COPY pyproject.toml ./
COPY rain/ ./rain/
COPY skills/ ./skills/
COPY scripts/ ./scripts/
COPY README.md LICENSE ./

# Install in editable mode minus the maturin rust build (rust extension
# is optional; pure-python path works without it). Pin numpy explicitly
# to avoid 2.x vs 1.x churn in transitive deps.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy>=1.26 scipy>=1.13 \
    && pip install --no-cache-dir --no-deps -e . 2>/dev/null || \
       pip install --no-cache-dir numpy scipy

# Default port + bind to all interfaces inside the container.
EXPOSE 8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/rain_net_serve.py", "--host", "0.0.0.0", "--port", "8080"]
