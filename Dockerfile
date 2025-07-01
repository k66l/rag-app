# Multi-stage Docker build for RAG Application API
# Stage 1: Build stage with all build dependencies
FROM python:3.11-slim-bullseye as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim-bullseye as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ragapp && useradd -r -g ragapp ragapp --create-home

# Create application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ ./app/
COPY start.sh ./
COPY pytest.ini ./
COPY run_tests.py ./

# Create necessary directories and set up model cache
RUN mkdir -p logs temp_docs /home/ragapp/.cache/huggingface && \
    chmod +x start.sh && \
    chown -R ragapp:ragapp /app /home/ragapp

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/home/ragapp/.cache/huggingface \
    HF_HOME=/home/ragapp/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/ragapp/.cache/sentence_transformers

# Switch to non-root user
USER ragapp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 