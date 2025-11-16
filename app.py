import os
import streamlit as st
from PIL import Image
import tempfile
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title='Food Detector', layout='centered')
st.title('Food Detector — Streamlit App (Ultralytics)')

# Behavior:
# - If WEIGHTS_PATH env var is set and file exists in repo, use it directly.
# - Else if DRIVE_FILE_ID env var is set, download the file from Google Drive using gdown.
# - Else if WEIGHTS_URL is set, try downloading that URL via requests.
# - The downloaded weights are stored in /tmp/weights (or app root) and cached across runs.

WEIGHTS_ENV_PATH = os.environ.get('WEIGHTS_PATH', 'best.pt')
DRIVE_FILE_ID = os.environ.get('DRIVE_FILE_ID', None)
WEIGHTS_URL = os.environ.get('WEIGHTS_URL', None)

WEIGHTS_LOCAL = Path('/tmp') / 'best.pt'  # primary cache location in Streamlit cloud
WEIGHTS_LOCAL2 = Path('best.pt')  # fallback repo root

def download_from_gdrive(file_id, out_path):
    try:
        import gdown
    except Exception:
        # install gdown if missing
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out_path), quiet=False)
    return out_path.exists()

def download_from_url(url, out_path):
    try:
        import requests
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
        import requests
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        return False
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path.exists()

@st.cache_resource
def prepare_weights():
    # 1) If WEIGHTS_PATH exists locally (in repo), use it
    repo_path = Path(WEIGHTS_ENV_PATH)
    if repo_path.exists():
        return str(repo_path.resolve())

    # 2) If we already downloaded to /tmp, use it
    if WEIGHTS_LOCAL.exists():
        return str(WEIGHTS_LOCAL)

    # 3) Try Google Drive
    if DRIVE_FILE_ID:
        st.info("Downloading weights from Google Drive...")
        ok = download_from_gdrive(DRIVE_FILE_ID, WEIGHTS_LOCAL)
        if ok:
            st.success("Weights downloaded from Google Drive.")
            return str(WEIGHTS_LOCAL)
        else:
            st.error("Failed to download weights from Google Drive.")

    # 4) Try WEIGHTS_URL
    if WEIGHTS_URL:
        st.info("Downloading weights from WEIGHTS_URL...")
        ok = download_from_url(WEIGHTS_URL, WEIGHTS_LOCAL)
        if ok:
            st.success("Weights downloaded from WEIGHTS_URL.")
            return str(WEIGHTS_LOCAL)
        else:
            st.error("Failed to download weights from WEIGHTS_URL.")

    # 5) fallback to repo root best.pt if exists
    if WEIGHTS_LOCAL2.exists():
        return str(WEIGHTS_LOCAL2.resolve())

    return None

weights_path = prepare_weights()
st.sidebar.write("Weights used:", weights_path if weights_path else "(none)")
if not weights_path:
    st.warning("No weights available. Set WEIGHTS_PATH, DRIVE_FILE_ID, or WEIGHTS_URL in app settings and redeploy.")
    st.stop()

# Load model lazily
@st.cache_resource
def load_model(p):
    from ultralytics import YOLO
    return YOLO(p)

try:
    model = load_model(weights_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)
    if st.button('Run detection'):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(tmp.name)
        with st.spinner('Running inference...'):
            try:
                results = model.predict(source=tmp.name, conf=0.35, imgsz=640)
                res = results[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    st.write('Detections:')
                    for b in boxes:
                        # b.cls and b.conf are tensors in some ultralytics versions
                        try:
                            cls = int(b.cls[0].item())
                        except Exception:
                            cls = int(b.cls)
                        try:
                            conf = float(b.conf[0].item())
                        except Exception:
                            conf = float(b.conf)
                        st.write(f'Class: {cls} — Conf: {conf:.2f}')
                    # show plotted image
                    try:
                        out_img_path = '/tmp/streamlit_result.png'
                        res.plot(save=True)  # may save internally depending on ultralytics version
                        # try to find saved file in working dir as fallback
                        if os.path.exists(out_img_path):
                            st.image(out_img_path, use_column_width=True)
                        else:
                            # render from numpy
                            im = getattr(res, 'orig_img', None)
                            if im is not None:
                                import matplotlib.pyplot as plt
                                plt.imshow(im[..., ::-1])
                                plt.axis('off')
                                st.pyplot(plt)
                    except Exception as e:
                        st.write('Could not render plotted image:', e)
                else:
                    st.write('No boxes found.')
            except Exception as e:
                st.error('Inference failed: ' + str(e))

# small info
st.sidebar.markdown('---')
st.sidebar.markdown('Set environment variables in Streamlit Cloud:')
st.sidebar.markdown('* `DRIVE_FILE_ID` — Google Drive file ID of your weights (public or shared).') 
st.sidebar.markdown('* `WEIGHTS_URL` — direct HTTP URL to weights (optional).')
st.sidebar.markdown('* `WEIGHTS_PATH` — path inside repo, e.g. `/app/best.pt` (optional).')
