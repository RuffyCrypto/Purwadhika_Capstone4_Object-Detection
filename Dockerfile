FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/render

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /home/render

RUN pip install --upgrade pip

# Install main dependencies including ultralytics
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow pyyaml opencv-python ultralytics

# Install CPU PyTorch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 torchvision==0.16.0

# FORCE NUMPY < 2 (MUST BE LAST TO AVOID ABI CRASHES)
RUN pip install --no-cache-dir "numpy<2"

# Clone YOLOv12 repo if not included
RUN if [ ! -d "./yolov12" ]; then git clone https://github.com/sunsmarterjie/yolov12.git yolov12; fi

RUN mkdir -p /home/render/models

ENV PORT 10000
EXPOSE 10000

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
