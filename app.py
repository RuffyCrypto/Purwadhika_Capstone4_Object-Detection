import os
import streamlit as st
from PIL import Image
import tempfile
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title='Food Detector', layout='centered')
st.title('Food Detector — Streamlit App (Ultralytics)')

WEIGHTS_ENV_PATH = os.environ.get('WEIGHTS_PATH', 'best.pt')
DRIVE_FILE_ID = os.environ.get('DRIVE_FILE_ID', None)
WEIGHTS_URL = os.environ.get('WEIGHTS_URL', None)
DEBUG_SHOW_ENV = os.environ.get('DEBUG_SHOW_ENV', '0')  # set to "1" to print envs to sidebar

WEIGHTS_LOCAL = Path('/tmp') / 'best.pt'
WEIGHTS_LOCAL2 = Path('best.pt')

def download_from_gdrive(file_id, out_path):
    try:
        import gdown
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    # gdown may fail on very large files unless shared properly; ensure file is 'Anyone with link' or public
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
    repo_path = Path(WEIGHTS_ENV_PATH)
    if repo_path.exists():
        return str(repo_path.resolve())

    if WEIGHTS_LOCAL.exists():
        return str(WEIGHTS_LOCAL)

    if DRIVE_FILE_ID:
        st.info("Downloading weights from Google Drive...")
        try:
            ok = download_from_gdrive(DRIVE_FILE_ID, WEIGHTS_LOCAL)
        except Exception as e:
            st.error(f"Download from Google Drive failed: {e}")
            ok = False
        if ok:
            st.success("Weights downloaded from Google Drive.")
            return str(WEIGHTS_LOCAL)
        else:
            st.error("Failed to download weights from Google Drive. Check DRIVE_FILE_ID and sharing settings.")
    if WEIGHTS_URL:
        st.info("Downloading weights from WEIGHTS_URL...")
        try:
            ok = download_from_url(WEIGHTS_URL, WEIGHTS_LOCAL)
        except Exception as e:
            st.error(f"Download from WEIGHTS_URL failed: {e}")
            ok = False
        if ok:
            st.success("Weights downloaded from WEIGHTS_URL.")
            return str(WEIGHTS_LOCAL)
        else:
            st.error("Failed to download weights from WEIGHTS_URL.")
    if WEIGHTS_LOCAL2.exists():
        return str(WEIGHTS_LOCAL2.resolve())
    return None

weights_path = prepare_weights()
if DEBUG_SHOW_ENV == "1":
    st.sidebar.write("DEBUG envs:")
    st.sidebar.write("WEIGHTS_ENV_PATH", WEIGHTS_ENV_PATH)
    st.sidebar.write("DRIVE_FILE_ID", DRIVE_FILE_ID)
    st.sidebar.write("WEIGHTS_URL", WEIGHTS_URL)
    st.sidebar.write("weights_path (resolved):", weights_path)

st.sidebar.write("Weights used:", weights_path if weights_path else "(none)")
if not weights_path:
    st.warning("No weights available. Set WEIGHTS_PATH, DRIVE_FILE_ID, or WEIGHTS_URL in app settings and redeploy.")
    st.stop()

@st.cache_resource
def load_model_safe(p):
    try:
        from ultralytics import YOLO
        return YOLO(p)
    except Exception as e:
        err = str(e)
        # Provide helpful message for libGL / opencv issues
        if "libGL.so.1" in err or "cannot open shared object file" in err or "GLIBC" in err:
            raise RuntimeError(
                "System dependency missing: libGL.so.1 (OpenGL). "
                "If you run this app locally or on Colab, install 'libgl1-mesa-glx' (Debian/Ubuntu): "
                "`sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx`.\\n\n"
                "On Streamlit Cloud, ensure requirements.txt contains 'opencv-python-headless' (remove 'opencv-python')."
            )
        else:
            raise RuntimeError(f"Failed to import/load ultralytics YOLO: {err}")

# Attempt to load model and show friendly error on failure
try:
    model = load_model_safe(weights_path)
except Exception as e:
    st.error("Failed to load model: " + str(e))
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
                # results is a list-like; take first
                res = results[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    st.write('Detections:')
                    for b in boxes:
                        try:
                            cls = int(b.cls[0].item())
                        except Exception:
                            cls = int(b.cls)
                        try:
                            conf = float(b.conf[0].item())
                        except Exception:
                            conf = float(b.conf)
                        st.write(f'Class: {cls} — Conf: {conf:.2f}')
                    # render image
                    try:
                        # attempt to plot using the result utilities
                        try:
                            # many ultralytics versions support res.plot()
                            img_plot = res.plot(save=False)
                            # res.plot may return numpy array or None; handle both
                            if img_plot is not None:
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(8,6))
                                plt.imshow(img_plot[..., ::-1])
                                plt.axis('off')
                                st.pyplot(fig)
                            else:
                                # fallback to orig_img
                                im = getattr(res, 'orig_img', None)
                                if im is not None:
                                    import matplotlib.pyplot as plt
                                    fig = plt.figure(figsize=(8,6))
                                    plt.imshow(im[..., ::-1])
                                    plt.axis('off')
                                    st.pyplot(fig)
                        except Exception:
                            im = getattr(res, 'orig_img', None)
                            if im is not None:
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(8,6))
                                plt.imshow(im[..., ::-1])
                                plt.axis('off')
                                st.pyplot(fig)
                    except Exception as e:
                        st.write('Could not render plotted image:', e)
                else:
                    st.write('No boxes found.')
            except Exception as e:
                st.error('Inference failed: ' + str(e))

st.sidebar.markdown('---')
st.sidebar.markdown('Set environment variables in Streamlit Cloud (Secrets):')
st.sidebar.markdown('Use TOML format, e.g.:')
st.sidebar.code('DRIVE_FILE_ID = \"1UcV5mF9fB1a-abcdefGHIJKLM\"')
st.sidebar.markdown('* `DRIVE_FILE_ID` — Google Drive file ID of your weights (public or shared).') 
st.sidebar.markdown('* `WEIGHTS_URL` — direct HTTP URL to weights (optional).')
st.sidebar.markdown('* `WEIGHTS_PATH` — path inside repo, e.g. `best.pt` (optional).')
st.sidebar.markdown('* `DEBUG_SHOW_ENV = \"1\"` — set to print debug envs in the sidebar.')