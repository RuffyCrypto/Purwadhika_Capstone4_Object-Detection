# Runtime-install workaround for headless OpenCV before other imports.
# This block runs pip at startup to ensure opencv-python-headless is installed
# to avoid libGL.so.1 errors on platforms where apt isn't available.
import sys, subprocess, os

# Force-install headless OpenCV and ultralytics early to avoid libGL issues.
# Startup will be slower because pip installs run at app start, but this
# is a robust fallback for environments like Streamlit Community Cloud.
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.7.0.72"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    # try without suppressing output if the pinned version fails
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])

# Ensure ultralytics present
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])

# Now import the rest of the libraries
import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path

st.set_page_config(page_title='Food Detector', layout='centered')
st.title('Food Detector — Streamlit App (Ultralytics)')

# Config via environment variables (set in Streamlit Secrets as TOML)
WEIGHTS_ENV_PATH = os.environ.get('WEIGHTS_PATH', 'best.pt')
DRIVE_FILE_ID = os.environ.get('DRIVE_FILE_ID', None)
WEIGHTS_URL = os.environ.get('WEIGHTS_URL', None)
DEBUG_SHOW_ENV = os.environ.get('DEBUG_SHOW_ENV', '0')  # set "1" to show debug envs

WEIGHTS_LOCAL = Path('/tmp') / 'best.pt'
WEIGHTS_LOCAL2 = Path('best.pt')

def download_from_gdrive(file_id, out_path):
    try:
        import gdown
    except Exception:
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
                    try:
                        # attempt to render plotted image
                        try:
                            img_plot = res.plot(save=False)
                            if img_plot is not None:
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(8,6))
                                plt.imshow(img_plot[..., ::-1])
                                plt.axis('off')
                                st.pyplot(fig)
                            else:
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