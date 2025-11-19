from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import tempfile, os, sys, subprocess, glob, base64
from pathlib import Path
from typing import Optional

app = FastAPI(title="YOLOv12 Inference Server - FoodCal")

# Configuration: set these via environment variables on Render or edit defaults
YOLOV12_DIR = os.environ.get("YOLOV12_DIR", "/home/render/yolov12")  # where repo lives in container
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/render/models/best.pt")  # default path to best.pt
CONF_DEFAULT = float(os.environ.get("CONF_DEFAULT", "0.25"))

@app.get("/healthz")
def healthz():
    return {"ok": True, "model_exists": Path(MODEL_PATH).exists()}

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: Optional[float] = Form(None)):
    conf_val = conf if conf is not None else CONF_DEFAULT

    # Save file to temp
    tmpdir = tempfile.mkdtemp()
    img_path = Path(tmpdir) / file.filename
    with open(img_path, "wb") as f:
        f.write(await file.read())

    # Ensure yolov12 directory and detect.py exist
    yolodir = Path(YOLOV12_DIR)
    detect_py = yolodir / "detect.py"
    if not yolodir.exists() or not detect_py.exists():
        return JSONResponse({"ok": False, "error": "yolov12_not_found", "path": str(YOLOV12_DIR)}, status_code=500)
    if not Path(MODEL_PATH).exists():
        return JSONResponse({"ok": False, "error": "model_not_found", "model_path": MODEL_PATH}, status_code=500)

    # Build command using python from environment
    python_exec = sys.executable
    cmd = [
        python_exec,
        str(detect_py),
        "--weights", str(MODEL_PATH),
        "--source", str(img_path),
        "--conf", str(conf_val),
        "--save-txt",
        "--save-img"
    ]

    # Run detect inside yolov12 directory so outputs go to yolov12/runs/detect/...
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(yolodir), env=os.environ.copy())
    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        return JSONResponse({"ok": False, "error": "detect_failed", "stdout": stdout, "stderr": stderr}, status_code=500)

    # find latest runs/detect/exp*
    runs = sorted(glob.glob(str(yolodir / "runs" / "detect" / "exp*")))
    if not runs:
        return JSONResponse({"ok": False, "error": "no_runs"}, status_code=500)
    latest = Path(runs[-1])

    annotated_img_path = latest / img_path.name
    label_file = latest / "labels" / (img_path.stem + ".txt")

    annotated_b64 = None
    if annotated_img_path.exists():
        with open(annotated_img_path, "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

    # parse labels (YOLO text format) to list
    detections = []
    if label_file.exists():
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    x_center = float(parts[1]); y_center = float(parts[2])
                    w = float(parts[3]); h = float(parts[4])
                    confs = float(parts[5]) if len(parts) > 5 else None
                    detections.append({"class": cls, "x": x_center, "y": y_center, "w": w, "h": h, "conf": confs})

    return {"ok": True, "stdout": stdout, "stderr": stderr, "annotated_image_b64": annotated_b64, "detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
