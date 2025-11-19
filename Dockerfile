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

# 1) Install NUMPY FIRST, pin < 2 to avoid ABI crashes with some libs
RUN pip install --no-cache-dir "numpy<2"

# 2) Install main dependencies, pin opencv-python to a version that works with numpy 1.x
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    pyyaml \
    "opencv-python<4.12" \
    ultralytics

# 3) Install CPU PyTorch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0 torchvision==0.16.0

# (Optional safety: ensure numpy<2 is still in place after all installs)
RUN pip install --no-cache-dir "numpy<2"

# --- HOTFIX: Patch Ultralytics AAttn qkv AttributeError ---
RUN python - << 'EOF'
from pathlib import Path
import ultralytics.nn.modules.block as block

path = Path(block.__file__)
text = path.read_text()

if "self.qkv(x)" in text:
    text = text.replace("self.qkv(x)", "self.qk(x)")
    path.write_text(text)
    print("Patched Ultralytics AAttn: self.qkv(x) -> self.qk(x)")
else:
    print("No AAttn qkv pattern found; skipping patch")
EOF
# --- END HOTFIX ---

# Clone YOLOv12 repo if not included
RUN if [ ! -d "./yolov12" ]; then git clone https://github.com/sunsmarterjie/yolov12.git yolov12; fi

RUN mkdir -p /home/render/models

ENV PORT 10000
EXPOSE 10000

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
