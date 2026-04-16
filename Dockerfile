# ===========================================================================
# Full GPU training image — NVIDIA B200 (Blackwell, sm_100, CUDA 12.8)
#
# Runs the 7-step perception pipeline on all training images, then fits the
# GraphClassifier on the resulting enriched detections + scene graphs.
#
# Build:
#   docker build -t ethackers-train .
#
# Run locally (GPU):
#   docker run --gpus all \
#     -e OPENAI_API_KEY="sk-..." \
#     -v /path/to/data:/data \
#     -v /path/to/output:/output \
#     ethackers-train --data-dir /data --output-dir /output --fast
# ===========================================================================

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

WORKDIR /app

# ── PyTorch + CUDA (largest layer — cached unless torch version changes) ─────
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ── HuggingFace (git install for SAM2 + Grounding DINO support) ─────────────
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers" \
    huggingface_hub accelerate

# ── All other Python dependencies ────────────────────────────────────────────
RUN pip install --no-cache-dir \
    numpy scipy \
    scikit-learn joblib \
    openai \
    tqdm psutil \
    pillow opencv-python-headless pycocotools \
    ultralytics matplotlib

# ── Copy application code (data excluded via .dockerignore) ──────────────────
COPY . .

# ── Default entrypoint ──────────────────────────────────────────────────────
ENTRYPOINT ["python", "scripts/train_action_model.py"]
CMD ["--fast"]
