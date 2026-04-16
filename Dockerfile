# ===========================================================================
# CPU-only training image — trains the GraphClassifier on pre-computed JSONs.
#
# No torch, no transformers, no CUDA. Just numpy + scikit-learn fitting on
# the enriched detection + scene graph JSONs that the pipeline already produced.
#
# Image size: ~200 MB.  Training time: seconds.
#
# Build:   docker build -t ethackers-train .
# Run:     docker run -v /path/to/pipeline_output:/data ethackers-train
# ===========================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy scipy scikit-learn joblib tqdm

COPY action_module/ ./action_module/
COPY src/label_ontology.py ./src/label_ontology.py
COPY scripts/train_action_model.py ./scripts/

ENTRYPOINT ["python", "scripts/train_action_model.py"]
