# ===========================================================================
# Training image — two entrypoints, one container.
#
# 1. Generate training data (GPU):
#      python scripts/generate_training_data.py --fast --n 500
#
# 2. Train classifier (CPU):
#      python scripts/train_action_model.py --det-dir /output/detections
#
# Build:   docker build -t ethackers-train .
#
# Run on Northflank/CoreWeave (GPU Job):
#   docker run --gpus all -e ANTHROPIC_API_KEY="sk-ant-..." \
#     -v $DATA:/data -v $OUTPUT:/output \
#     ethackers-train \
#     python scripts/generate_training_data.py \
#       --image-dir /data/images/train --output-dir /output --fast --n 500
# ===========================================================================

# Base image already ships with Python 3.11, PyTorch 2.8 + CUDA 12.8,
# git, curl, libgl — no reliance on Ubuntu apt mirrors for those.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Best-effort apt install for optional extras (opencv runtime libs).
# Wrapped in `|| true` so a temporary Ubuntu mirror outage doesn't break the
# build — the PyTorch base image already provides everything Python needs.
RUN (apt-get update && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 ca-certificates \
       && rm -rf /var/lib/apt/lists/*) || \
    echo "Apt install skipped (mirrors unreachable) — base image should cover essentials."

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install gsutil via pip (published by Google on PyPI, no apt / curl needed)
RUN pip install --no-cache-dir gsutil
RUN gsutil version

# HuggingFace transformers (SAM2 + Grounding DINO support in >=4.46).
# Using PyPI release instead of git master to avoid GitHub fetch issues
# on Northflank's build node.
RUN pip install --no-cache-dir \
    "transformers>=4.46" \
    huggingface_hub accelerate

# All other Python deps (PyTorch + torchvision already in base image)
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn joblib \
    anthropic pyyaml openai tqdm psutil \
    pillow opencv-python-headless pycocotools \
    ultralytics matplotlib

COPY . .
