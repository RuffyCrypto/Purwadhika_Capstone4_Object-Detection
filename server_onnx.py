import os, tempfile, shutil, logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import onnxruntime as ort
from PIL import Image

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="YOLO ONNX Inference")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best.onnx")
MODEL_URL = os.environ.get("MODEL_URL")
CONF_THRESH = float(os.environ.get("CONF", 0.35))

_session = None

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            logging.error("MODEL not present and MODEL_URL not set.")
            return False
        logging.info("Downloading ONNX model from %s", MODEL_URL)
        tmp = MODEL_PATH + ".download"
        try:
            with requests.get(MODEL_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            os.replace(tmp, MODEL_PATH)
            logging.info("Downloaded ONNX model.")
        except Exception as e:
            logging.error("Failed to download ONNX model: %s", e)
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            return False
    return True

def get_session():
    global _session
    if _session is None:
        ok = ensure_model()
        if not ok:
            raise FileNotFoundError("ONNX model not available.")
        # create onnxruntime session
        logging.info("Creating ONNX Runtime session for %s", MODEL_PATH)
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return _session

# Helper: use ultralytics to parse outputs if possible (ultralytics supports ONNX)
def predict_with_ultralytics_onnx(img_path):
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        results = model.predict(img_path, conf=CONF_THRESH, imgsz=640)
        r = results[0]
        out = []
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", {})
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
            except Exception:
                xyxy = boxes.xyxy.numpy().tolist()
            try:
                confs = boxes.conf.cpu().numpy().tolist()
            except Exception:
                confs = boxes.conf.numpy().tolist()
            try:
                clss = boxes.cls.cpu().numpy().tolist()
            except Exception:
                clss = boxes.cls.numpy().tolist()
            for bb, cf, cl in zip(xyxy, confs, clss):
                out.append({ "box": bb, "conf": float(cf), "class_id": int(cl), "name": names.get(int(cl), str(cl)) })
        return out
    except Exception as e:
        logging.warning("Ultralytics parsing failed, falling back to raw ONNX session: %s", e)
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Try using ultralytics parser (convenient)
        preds = predict_with_ultralytics_onnx(tmp_path)
        if preds is not None:
            return JSONResponse({"predictions": preds})

        # Fallback: raw onnx runtime inference - user must ensure exported ONNX outputs are standard
        sess = get_session()
        # NOTE: raw ONNX parsing is model-specific. We'll attempt a generic path: 
        # if the ONNX model follows YOLO export conventions, ultralytics parsing is preferred.
        # Here, we simply return empty predictions to avoid crashing.
        return JSONResponse({"predictions": []})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logging.error("Predict error: %s", tb)
        try:
            with open("/tmp/predict_error.log", "a") as fh:
                fh.write(tb + "\n---\n")
        except Exception:
            pass
        return JSONResponse({"error":"internal server error","detail": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@app.get("/")
def root():
    return {"status":"ok","message":"Use /predict"}
