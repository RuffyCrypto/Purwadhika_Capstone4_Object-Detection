FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/render

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your FastAPI app and other project files
COPY . /home/render

# Clone YOLOv12 repo (for detect.py, configs, etc.)
RUN if [ ! -d "./yolov12" ]; then git clone https://github.com/sunsmarterjie/yolov12.git yolov12; fi

RUN pip install --upgrade pip

# 1) Core numeric + torch stack
RUN pip install --no-cache-dir "numpy<2"

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0 torchvision==0.16.0

# 2) Other libs your API and YOLOv12 need
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    pyyaml \
    "opencv-python<4.12" \
    thop \
    seaborn

# 3) Install YOLOv12 fork of Ultralytics from GitHub (recommended method)
#    This overwrites any existing 'ultralytics' with the YOLOv12 version.
RUN pip install --no-cache-dir "git+https://github.com/sunsmarterjie/yolov12.git"

RUN mkdir -p /home/render/models

ENV PORT 10000
EXPOSE 10000

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
