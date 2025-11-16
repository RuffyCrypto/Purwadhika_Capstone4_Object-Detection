FROM python:3.11-slim

# Install system deps including libGL and ffmpeg (for ultralytics/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_server.txt /app/requirements_server.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements_server.txt

# Copy server code
COPY server_fastapi.py /app/server_fastapi.py

# On container start: if MODEL_URL env var present, download to /app/best.pt;
# otherwise, require that best.pt is present in image or mounted.
CMD bash -lc '\
  if [ ! -f /app/best.pt ] && [ -n \"$MODEL_URL\" ]; then \
    echo \"Downloading model from $MODEL_URL\"; \
    curl -L --max-redirs 5 --retry 5 --retry-delay 2 \"$MODEL_URL\" -o /app/best.pt; \
  fi; \
  exec uvicorn server_fastapi:app --host 0.0.0.0 --port ${PORT:-8000}'