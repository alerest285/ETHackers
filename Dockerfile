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

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl wget libgl1-mesa-glx libglib2.0-0 \
    apt-transport-https ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (provides gsutil) — separate layer for clarity
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/* \
    && gsutil --version \
    && gcloud --version

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

WORKDIR /app

# PyTorch + CUDA (cached layer)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# HuggingFace transformers from git (SAM2 + Grounding DINO support)
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers" \
    huggingface_hub accelerate

# All other deps
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn joblib \
    anthropic pyyaml openai tqdm psutil \
    pillow opencv-python-headless pycocotools \
    ultralytics matplotlib

COPY . .
