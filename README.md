YOLO ONNX Inference Server
==========================

This repo runs a simple FastAPI server for ONNX models. It prefers using Ultralytics to parse
ONNX outputs (if ultralytics can load the ONNX), otherwise falls back to an empty response
to avoid crashing - you can implement raw ONNX output decoding if needed.

Usage:
- Build and run Docker, or deploy to Render. Set MODEL_URL in Render environment variables
  to a direct URL of your `best.onnx` file.

Example Docker build:
  docker build -t yolo-onnx-server:latest .
  docker run -e MODEL_URL="https://.../best.onnx" -p 8000:8000 yolo-onnx-server:latest
