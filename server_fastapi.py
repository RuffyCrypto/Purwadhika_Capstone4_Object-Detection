import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile, uvicorn
from typing import List, Dict, Any

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best.pt")
MODEL_URL = os.environ.get("MODEL_URL", None)  # public HTTP URL to download weights if not baked in
CONF_THRESH = float(os.environ.get("CONF", 0.35))

app = FastAPI(title="YOLO Inference API")
_model = None

def ensure_model():
    # If model file does not exist and MODEL_URL provided, download it
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        try:
            import requests, shutil
            tmp = MODEL_PATH + ".download"
            with requests.get(MODEL_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            os.replace(tmp, MODEL_PATH)
            print(f"Downloaded MODEL_URL to {MODEL_PATH}")
        except Exception as e:
            print("Failed to download MODEL_URL:", e)
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
    return os.path.exists(MODEL_PATH)

def get_model():
    global _model
    if _model is None:
        # ensure model present (attempt download if needed)
        ok = ensure_model()
        if not ok:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} and MODEL_URL not set or download failed.")
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model

@app.on_event("startup")
def on_startup():
    # attempt to pre-load model (optional) to fail early if missing
    try:
        get_model()
        print("Model loaded on startup.")
    except Exception as e:
        # print warning; keep server alive to allow later download via endpoint
        print("Warning: model not loaded on startup:", e)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        model = get_model()
        results = model.predict(source=tmp_path, conf=CONF_THRESH, imgsz=640)
        r = results[0]
        out = []
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", {})
        if boxes is not None:
            # extract arrays safely
            try:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
            except Exception:
                try:
                    xyxy = boxes.xyxy.numpy().tolist()
                except Exception:
                    xyxy = []
            try:
                confs = boxes.conf.cpu().numpy().tolist()
            except Exception:
                try:
                    confs = boxes.conf.numpy().tolist()
                except Exception:
                    confs = [None]*len(xyxy)
            try:
                clss = boxes.cls.cpu().numpy().tolist()
            except Exception:
                try:
                    clss = boxes.cls.numpy().tolist()
                except Exception:
                    clss = [None]*len(xyxy)

            for bb, cf, cl in zip(xyxy, confs, clss):
                label = names.get(int(cl), str(int(cl))) if cl is not None else str(cl)
                out.append({
                    "box": bb,
                    "conf": float(cf) if cf is not None else None,
                    "class_id": int(cl) if cl is not None else None,
                    "name": label
                })
        return JSONResponse({"predictions": out})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
