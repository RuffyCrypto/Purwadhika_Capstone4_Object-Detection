import io
import requests
from PIL import Image
import streamlit as st

# =====================
# CONFIG
# =====================
API_URL = "https://purwadhika-capstone4-object-detection.onrender.com/detect"

st.set_page_config(
    page_title="Capstone 4 – Object Detection",
    page_icon="🧠",
    layout="centered",
)

st.title("🍜 Capstone 4 – Makanan Detection")
st.write("Upload gambar makanan, nanti model YOLOv12 di Render yang inferensi.")

# =====================
# SIDEBAR
# =====================
st.sidebar.header("Inference Settings")

conf = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.99,
    value=0.5,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.write("Backend API:")
st.sidebar.code(API_URL, language="text")

# =====================
# MAIN UI
# =====================
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run detection"):
        with st.spinner("Sending to backend and running detection..."):
            try:
                # prepare multipart/form-data for backend
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                }
                data = {
                    "conf": str(conf),
                }
                resp = requests.post(API_URL, files=files, data=data, timeout=120)

                if resp.status_code != 200:
                    st.error(f"Backend returned status code {resp.status_code}")
                    st.text(resp.text)
                else:
                    result = resp.json()

                    # Show raw JSON
                    st.subheader("Raw response JSON")
                    st.json(result)

                    if result.get("ok"):
                        st.success("Detection succeeded ✅")

                        # Try to show some helpful text from stdout
                        stdout = result.get("stdout", "")
                        if stdout:
                            st.subheader("Backend stdout")
                            st.code(stdout)

                        # OPTIONAL: if someday you modify backend to return
                        # an image URL or base64, you can display it here.
                        # For now, only stdout + JSON.

                    else:
                        st.error("Detection failed ❌")
                        stderr = result.get("stderr", "")
                        if stderr:
                            st.subheader("Backend stderr")
                            st.code(stderr)
            except Exception as e:
                st.error(f"Error calling backend: {e}")
