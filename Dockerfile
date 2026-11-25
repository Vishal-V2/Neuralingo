# Neuralingo: translates the manga pages in /app/out into /app/output.
#
# Build (from the project folder):
#   docker build -t neuralingo .                              # NVIDIA GPU (falls back to CPU)
#   docker build -t neuralingo:cpu --build-arg TORCH_VARIANT=cpu .   # CPU only, much smaller
#
# Run: put pages in ./pages, translations appear in ./translated:
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" \
#     -e GEMINI_API_KEY=... \
#     -v "$PWD/pages:/app/out:ro" -v "$PWD/translated:/app/output" neuralingo
# (leave out --gpus all without an NVIDIA GPU; see DOCKER.md)

FROM python:3.12-slim

# cuda: PyPI's torch build, with CUDA (runs on CPU too); cpu: CPU-only build.
ARG TORCH_VARIANT=cuda
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # No display in a container: Qt (used to draw the English text) renders offscreen.
    QT_QPA_PLATFORM=offscreen \
    # Model caches baked into the image, readable by any user (docker run --user).
    HF_HOME=/opt/cache/huggingface \
    TORCH_HOME=/opt/cache/torch \
    LAMA_MODEL=/opt/models/big-lama.pt \
    # Writable scratch for libraries that keep settings in the home folder.
    HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Libraries Qt and OpenCV load at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 libxkbcommon0 libfontconfig1 libdbus-1-3 fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}; \
    fi

# Exactly the versions of the working development environment, installed as they are:
# some packages declare outdated limits (simple-lama-inpainting: Pillow < 10) that the
# working set ignores, so pip must not re-resolve them.
COPY requirements.docker.txt .
RUN pip install --no-deps -r requirements.docker.txt

# Download the models now, so the first run doesn't (manga-ocr from Hugging Face,
# LaMa inpainting from GitHub; the YOLO and MiGAN weights are in the repository).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('kha-white/manga-ocr-base')" \
    && mkdir -p /opt/models \
    && curl -fL -o /opt/models/big-lama.pt \
        https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
    && chmod -R a+rX /opt/cache /opt/models

COPY . .
# Default settings (no keys; pass them as environment variables, or mount a config.yaml).
RUN cp config.example.yaml config.yaml \
    && mkdir -p out output \
    && chmod -R a+rwX out output

CMD ["python", "app.py"]
