    # Streamlit Client for Food Detector

This repo contains a lightweight Streamlit client that sends uploaded images to
an inference server and displays detected food items and estimated calories.

## Usage
1. Set `INFERENCE_API_URL` in Streamlit Secrets (TOML) or as an env var locally.

    Example (TOML in Streamlit Cloud):
    ```toml
    INFERENCE_API_URL = "https://your-server.onrender.com/predict"
    ```

2. Deploy to Streamlit Cloud or run locally:
    ```bash
    pip install -r requirements.txt
    INFERENCE_API_URL="https://.../predict" streamlit run streamlit_client.py
    ```
