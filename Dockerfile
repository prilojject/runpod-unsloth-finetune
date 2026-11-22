# Unsloth LoRA/QLoRA fine-tuning worker for Runpod Serverless.
# Base: official PyTorch runtime image (CUDA 12.6 -> runs on drivers for CUDA 12.6 .. 13.x).
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    UNSLOTH_DISABLE_STATISTICS=1

# Triton (used by Unsloth kernels) compiles small C launchers at runtime -> needs a C compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the base image's torch stack: freeze it as constraints so pip cannot swap it.
COPY requirements.txt /app/requirements.txt
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && pip freeze > /app/requirements.lock \
    && python -c "import torch, trl, transformers, peft, datasets, runpod; print('torch', torch.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| runpod', runpod.__version__)"

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked into the image: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
CMD ["python", "-u", "/app/handler.py"]
