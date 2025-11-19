#!/usr/bin/env bash
set -e
if [ -n "$MODEL_URL" ]; then
  mkdir -p /home/render/models
  echo "Downloading model from $MODEL_URL"
  wget -O /home/render/models/best.pt "$MODEL_URL" || { echo "Model download failed"; exit 0; }
fi
exec "$@"
