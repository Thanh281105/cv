FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG EAGLE_REVISION=main

ENV EAGLE_ROOT=/opt/Eagle/Embodied
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV WANDB_DISABLED=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-locany.txt /workspace/requirements-locany.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /workspace/requirements-locany.txt

RUN git clone https://github.com/NVlabs/Eagle.git /opt/Eagle \
    && cd /opt/Eagle \
    && git checkout "${EAGLE_REVISION}" \
    && python3 -m pip install -e /opt/Eagle/Embodied

COPY viground /workspace/viground
COPY scripts /workspace/scripts
COPY configs /workspace/configs

ENV PYTHONPATH=/workspace:/opt/Eagle/Embodied:${PYTHONPATH}
WORKDIR /workspace

