# Multi-stage build for Artificial Memory Operator
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY k8s/operator/ ./k8s/operator/
COPY k8s/crds/ ./k8s/crds/

# Install the package with operator dependencies
RUN pip install --no-cache-dir -e ".[operator]"

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 operator

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Change ownership
RUN chown -R operator:operator /app

USER operator

# Health check endpoint
EXPOSE 8080

# Run operator
ENTRYPOINT ["python", "-m", "k8s.operator"]