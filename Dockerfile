FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Avoid interactive tzdata prompts during image builds.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

WORKDIR /workspace

# Install runtime system libraries.
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Apply the configured timezone.
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install the pinned Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install "numpy<2.0.0"

RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-build-isolation

CMD ["/bin/bash"]
