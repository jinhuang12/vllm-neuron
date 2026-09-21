FROM ubuntu:24.04

# Keep the host's Python ABI. The Neuron venv and SDK are mounted read-only.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates python3.12 libpython3.12t64 \
       libarchive13t64 libatomic1 libgfortran5 libgomp1 libnuma1 libtinfo6 libxml2 libzstd1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_LOGICAL_NC_CONFIG=2

CMD ["python3.12", "--version"]
