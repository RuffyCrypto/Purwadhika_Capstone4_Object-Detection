import streamlit as st
from PIL import Image
import requests, io, os
from collections import Counter, defaultdict

st.set_page_config(page_title="Food Detector (Client)", layout="centered")
st.title("Food Detector — Client")

# read API URL from secrets; fallback to env var for local testing
API_URL = None
try:
    API_URL = st.secrets["INFERENCE_API_URL"]
except Exception:
    API_URL = os.environ.get("INFERENCE_API_URL")

if not API_URL:
    st.error("INFERENCE_API_URL not set. Set it in Streamlit Secrets (TOML) or INFERENCE_API_URL env var.")
    st.stop()

if API_URL.endswith("/"):
    API_URL = API_URL.rstrip("/")
predict_url = API_URL  # should point to /predict

st.sidebar.write("Inference API:", predict_url)

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image of food to analyze.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, use_column_width=True)

if st.button("Run detection"):
    with st.spinner("Calling inference server..."):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            resp = requests.post(predict_url, files=files, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            st.error("API call failed: " + str(e))
            st.stop()

    data = resp.json()
    preds = data.get("predictions", [])
    if not preds:
        st.info("No detections returned by server.")
        st.stop()

    # --- Calorie mapping: adjust names exactly as your model/class names.
    calorie_db = {
        "Ayam Goreng -260 kal per 100 gr-": {"kcal": 260, "unit":"100g"},
        "Capcay -67 kal per 100gr-": {"kcal": 67, "unit":"100g"},
        "Nasi -129 kal per 100gr-": {"kcal": 129, "unit":"100g"},
        "Sayur bayam -36 kal per 100gr-": {"kcal": 36, "unit":"100g"},
        "Sayur kangkung -98 kal per 100gr-": {"kcal": 98, "unit":"100g"},
        "Sayur sop -22 kal per 100gr-": {"kcal": 22, "unit":"100g"},
        "Tahu -80 kal per 100 gr-": {"kcal": 80, "unit":"100g"},
        "Telur Dadar -93 kal per 100gr-": {"kcal": 93, "unit":"100g"},
        "Telur Mata Sapi -110kal1butir-": {"kcal": 110, "unit":"item"},
        "Telur Rebus -78kal 1butir-": {"kcal": 78, "unit":"item"},
        "Tempe -225 kal per 100 gr-": {"kcal": 225, "unit":"100g"},
        "Tumis buncis -65 kal per 100gr-": {"kcal": 65, "unit":"100g"},
    }

    # Interpret predictions and compute totals
    counts = Counter()
    details = []
    total_kcal = 0.0

    for p in preds:
        name = p.get("name") or str(p.get("class_id", "unknown"))
        conf = p.get("conf", 0.0) or 0.0
        box = p.get("box", None)
        counts[name] += 1
        # lookup calorie info (best-effort matching)
        info = calorie_db.get(name)
        kcal_val = 0.0
        unit = None
        if info:
            kcal_val = float(info["kcal"])
            unit = info["unit"]
            # assumption: each detection corresponds to the same gram/unit as in DB.
            if unit == "100g" or unit == "item":
                total_kcal += kcal_val
            else:
                total_kcal += kcal_val
        else:
            # attempt substring matching if exact key not found
            matched = False
            for k, v in calorie_db.items():
                if k.split()[0].lower() in name.lower():
                    kcal_val = float(v["kcal"])
                    total_kcal += kcal_val
                    unit = v["unit"]
                    matched = True
                    break
            if not matched:
                # unknown food -> 0 kcal by default
                kcal_val = 0.0
        details.append({"name": name, "conf": conf, "box": box, "kcal_per_detect": kcal_val, "unit": unit})

    # show details
    st.subheader("Detections")
    for d in details:
        st.write(f"- **{d['name']}** — conf {d['conf']:.2f} — kcal per detection: {d['kcal_per_detect']} ({d['unit']})")

    st.markdown("---")
    st.metric("Total Calories", f"{int(total_kcal)} kcal")

    # Optional: group counts
    st.subheader("Summary")
    for name, c in counts.items():
        st.write(f"{name}: {c}")

    st.info("Tip: if you want to render bounding boxes on the image from server, modify server to return plotted image bytes.")
