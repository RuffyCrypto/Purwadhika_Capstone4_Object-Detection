# YOLO Inference Server (Docker + MODEL_URL)

This repository runs a FastAPI server that loads an Ultralytics YOLO model (`best.pt`) and exposes an inference endpoint `/predict`.

**Behavior**:
- If `/app/best.pt` exists inside the container (baked in or mounted), it will be used.
- Else, if environment variable `MODEL_URL` is set to a public HTTP(S) URL pointing to the weights (e.g., S3, GCS, GitHub Release raw link), the container will download it at startup to `/app/best.pt` and use it.

## Files
- `server_fastapi.py` - FastAPI server
- `requirements_server.txt` - Python dependencies
- `Dockerfile` - Docker image (installs libGL)

## Build & Run locally (example)

Build:
```bash
docker build -t yolo-inference:latest .
```

Run (with MODEL_URL to download weights at container start):
```bash
docker run -e MODEL_URL="https://your-bucket/path/to/best.pt" -p 8000:8000 yolo-inference:latest
```

Or, mount weights into container:
```bash
docker run -v /path/to/best.pt:/app/best.pt -p 8000:8000 yolo-inference:latest
```

## Deploy to Render (example)
- Create new Web Service → Connect to this repo.
- Choose "Docker" as the environment (Render will build the Dockerfile).
- Set environment variables in Render: `MODEL_URL` (public URL) and optionally `PORT`.

## Notes
- Prefer storing large weights on S3/GCS and using pre-signed URLs (MODEL_URL) for secure access.
- If using GitHub Releases, use the raw URL for the release asset (beware rate limits).
- For GPU acceleration, deploy to a GPU-enabled host and adapt the base image accordingly (e.g., nvidia/cuda images) and install proper CUDA & torch wheels.
