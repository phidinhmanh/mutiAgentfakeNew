# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen: Use the exact versions from uv.lock
# --no-dev: Skip development dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create a non-root user for security
RUN useradd -m appuser && \
    mkdir -p /app/models_cache && \
    chown -R appuser:appuser /app
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI server using uvicorn
CMD ["uv", "run", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
