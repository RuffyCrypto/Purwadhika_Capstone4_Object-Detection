FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux;         apt-get update || (apt-get update --allow-releaseinfo-change);         apt-get install -y --no-install-recommends           ca-certificates           curl           gnupg           ffmpeg           libgl1-mesa-glx           git         || apt-get install -y --no-install-recommends --fix-missing libgl1-mesa-glx ffmpeg curl ca-certificates git;         rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_server.txt /app/requirements_server.txt

RUN pip install --no-cache-dir -r /app/requirements_server.txt

COPY server_fastapi.py /app/server_fastapi.py

CMD bash -lc '\
  if [ ! -f /app/best.pt ] && [ -n "$MODEL_URL" ]; then \
    echo "Downloading model from $MODEL_URL"; \
    curl -L --max-redirs 5 --retry 5 --retry-delay 2 "$MODEL_URL" -o /app/best.pt; \
  fi; \
  exec uvicorn server_fastapi:app --host 0.0.0.0 --port ${PORT:-8000}'
