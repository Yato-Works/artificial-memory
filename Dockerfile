# Dockerfile for Artificial Memory / Context Runtime

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[web,llm]"

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create runtime data directories
RUN mkdir -p /app/memory_files /app/data && chown -R appuser:appuser /app/memory_files /app/data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from artificial_memory.storage.sqlite_store import SQLiteMemoryStore; s = SQLiteMemoryStore('data/memory.db'); exit(0 if s.health_check() else 1)"

# Run the application
CMD ["uvicorn", "artificial_memory.api.server:app", "--host", "0.0.0.0", "--port", "8000"]