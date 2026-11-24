# ==============================================================================
# SmartCell AI Blood Cell Screening — Production Container for Render Free Tier
# Multi-Disease Architecture: Mode 1 (Malaria) & Mode 2 (ALL Leukemia)
# ==============================================================================

FROM python:3.11-slim

LABEL maintainer="SmartCell Team"
LABEL description="SmartCell AI Blood Cell Screening Backend for Render Free Tier"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    HOST=0.0.0.0 \
    SMARTCELL_ROOT=/app \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install minimal OS dependencies for OpenCV, PIL, and health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies optimized for Render CPU
COPY deployment/render_backend/requirements-render.txt ./requirements-render.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-render.txt

# Copy application source code and scripts
COPY scripts/ /app/scripts/
COPY deployment/render_backend/ /app/deployment/render_backend/

# Copy trained AI model checkpoints
COPY models/ /app/models/
COPY results/yolo_malaria_run/weights/best.pt /app/results/yolo_malaria_run/weights/best.pt

# Copy benchmark preset evaluation sets for demonstration
COPY yolo_malaria/images/test/ /app/yolo_malaria/images/test/
COPY yolo_malaria/labels/test/ /app/yolo_malaria/labels/test/
COPY dataset_all/test/ /app/dataset_all/test/

# Create runtime directories
RUN mkdir -p /app/results/demo_app_runs/_uploads \
             /app/results/smartcell_pipeline \
             /app/results/all_idb \
             /app/results/metrics

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8501}/api/health || exit 1

# Launch SmartCell Render backend server
CMD ["sh", "-c", "python deployment/render_backend/render_app.py --host 0.0.0.0 --port ${PORT:-8501}"]
