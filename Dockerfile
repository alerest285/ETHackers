# ===========================================================================
# Navsense — GPU image for Northflank / CoreWeave B200
#
# Default: serves the FastAPI web UI (app.py) on port 7860.
# Override CMD to run training or batch jobs, e.g.:
#   python scripts/generate_training_data.py --fast --n 500
#   python scripts/train_action_model.py --det-dir /output/detections
#
# Build:   docker build -t navsense .
# Run:     docker run --gpus all -p 7860:7860 \
#            -e ANTHROPIC_API_KEY="sk-ant-..." \
#            -e OPENAI_API_KEY="sk-..." \
#            navsense
#
# Northflank:
#   - Service type: Deployment (or Combined)
#   - Resources: 4 vCPU, 16 GB RAM, 1× B200 (2× for bigger headroom)
#   - Ephemeral storage: 20 GB (HF caches models on first run)
#   - Port: 7860  (HTTP, public)
#   - SHM size: 2 GB
#   - Env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY
# ===========================================================================

# PyTorch 2.8 + CUDA 12.8 base (ships Python 3.11, git, curl, libgl).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Best-effort apt install for optional extras (opencv runtime libs).
RUN (apt-get update && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 ca-certificates \
       && rm -rf /var/lib/apt/lists/*) || \
    echo "Apt install skipped (mirrors unreachable) — base image covers essentials."

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Optional: Google Cloud Storage CLI for dataset/model transfer
RUN pip install --no-cache-dir gsutil

# HuggingFace stack (transformers >=4.46 has SAM2 + Grounding DINO support).
RUN pip install --no-cache-dir \
    "transformers>=4.46" \
    huggingface_hub accelerate safetensors

# Web server — FastAPI + uvicorn + multipart form parsing for uploads
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart

# Pipeline deps (PyTorch + torchvision already in base image)
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn joblib \
    anthropic openai pyyaml tqdm psutil \
    pillow opencv-python-headless pycocotools \
    ultralytics matplotlib

# Copy the entire project (models download on first request to /run_full)
COPY . .

# Pre-download the smaller DepthAnything weights at build time so the first
# request doesn't wait on ~200 MB of model weights. Best-effort — if the HF
# hub is unreachable at build time, the container will just lazy-load on
# the first request instead.
RUN python -c "from transformers import pipeline; \
    p = pipeline(task='depth-estimation', \
                 model='depth-anything/Depth-Anything-V2-Small-hf'); \
    print('Depth model cached')" || echo "Depth prefetch skipped"

EXPOSE 7860

# Default: launch the web UI. Override on Northflank to run training jobs.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
