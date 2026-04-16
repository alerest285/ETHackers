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

# Best-effort apt install for optional extras (opencv + gnupg for gcloud).
# Wrapped in `|| true` so a temporary Ubuntu mirror outage doesn't break the
# build — the PyTorch base image already provides everything Python needs.
RUN (apt-get update && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 gnupg ca-certificates wget \
       && rm -rf /var/lib/apt/lists/*) || \
    echo "Apt install skipped (mirrors unreachable) — base image should cover essentials."

# Install Google Cloud SDK via curl installer (downloads from
# dl.google.com, independent of Ubuntu mirrors).
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcloud-install.sh \
    && bash /tmp/gcloud-install.sh --disable-prompts --install-dir=/usr/local \
    && rm /tmp/gcloud-install.sh
ENV PATH=/usr/local/google-cloud-sdk/bin:$PATH
RUN gsutil --version && gcloud --version

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# HuggingFace transformers from git (SAM2 + Grounding DINO support)
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers" \
    huggingface_hub accelerate

# All other Python deps (PyTorch + torchvision already in base image)
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn joblib \
    anthropic pyyaml openai tqdm psutil \
    pillow opencv-python-headless pycocotools \
    ultralytics matplotlib

COPY . .
