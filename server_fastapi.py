import os
import tempfile
import traceback
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best.pt")
MODEL_URL = os.environ.get("MODEL_URL")
CONF_THRESH = float(os.environ.get("CONF", 0.35))

app = FastAPI(title="YOLO Inference API")
_model = None

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            logging.error("MODEL_PATH missing and MODEL_URL not provided.")
            return False

        logging.info(f"Downloading model from {MODEL_URL}")
        try:
            import requests, shutil
            tmp = MODEL_PATH + ".download"

            with requests.get(MODEL_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            os.replace(tmp, MODEL_PATH)
            logging.info(f"Downloaded MODEL_URL to {MODEL_PATH}")

        except Exception as e:
            logging.error(f"Failed to download MODEL_URL: {e}")
            return False

    return True

def get_model():
    global _model
    if _model is None:
        ok = ensure_model()
        if not ok:
            raise FileNotFoundError("Model not available and download failed.")

        from ultralytics import YOLO
        logging.info("Loading YOLO model...")
        _model = YOLO(MODEL_PATH)
        logging.info("Model loaded.")

    return _model

@app.on_event("startup")
def on_startup():
    try:
        get_model()
        logging.info("Model loaded on startup.")
    except Exception as e:
        logging.error(f"Warning: model not loaded on startup: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = get_model()
        results = model.predict(source=tmp_path, conf=CONF_THRESH, imgsz=640)
        r = results[0]

        predictions = []
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", {})

        if boxes is not None:
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
                    confs = [None] * len(xyxy)

            try:
                clss = boxes.cls.cpu().numpy().tolist()
            except Exception:
                try:
                    clss = boxes.cls.numpy().tolist()
                except Exception:
                    clss = [None] * len(xyxy)

            for bb, cf, cl in zip(xyxy, confs, clss):
                label = names.get(int(cl), str(cl)) if cl is not None else "unknown"

                predictions.append({
                    "box": bb,
                    "conf": float(cf) if cf is not None else None,
                    "class_id": int(cl) if cl is not None else None,
                    "name": label
                })

        return JSONResponse({"predictions": predictions})

    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Predict handler exception:\n" + tb)

        try:
            with open("/tmp/predict_error.log", "a") as f:
                f.write(tb + "\n---\n")
        except:
            pass

        return JSONResponse(
            {"error": "internal server error", "detail": str(e)},
            status_code=500,
        )

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.get("/")
def root():
    return {"status": "ok", "message": "Use /predict"}
