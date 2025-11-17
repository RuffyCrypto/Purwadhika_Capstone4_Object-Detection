FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux;         apt-get update || (apt-get update --allow-releaseinfo-change);         apt-get install -y --no-install-recommends           ca-certificates           curl           gnupg           ffmpeg           libgl1-mesa-glx           git         || apt-get install -y --no-install-recommends --fix-missing libgl1-mesa-glx ffmpeg curl ca-certificates git;         rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server_onnx.py /app/server_onnx.py

CMD bash -lc '\
  if [ ! -f /app/best.onnx ] && [ -n "$MODEL_URL" ]; then \
    echo "Downloading ONNX model from $MODEL_URL"; \
    curl -L --max-redirs 5 --retry 5 --retry-delay 2 "$MODEL_URL" -o /app/best.onnx; \
  fi; \
  exec uvicorn server_onnx:app --host 0.0.0.0 --port ${PORT:-8000}'
