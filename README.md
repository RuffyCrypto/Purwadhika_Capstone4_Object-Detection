# Purwadhika_Capstone4_Object-Detection_AIEngineering

# Food Detector Streamlit App (auto-download weights)

This repository contains a Streamlit app that loads YOLO weights (Ultralytics) and runs inference on uploaded images.
It supports downloading weights from Google Drive by setting the `DRIVE_FILE_ID` environment variable in Streamlit Cloud.

Files:
- app.py — Streamlit app (auto-downloads weights if needed)
- requirements.txt — Python dependencies

Deployment (Streamlit Community Cloud):
1. Push this repo to GitHub.
2. Create a new app on https://share.streamlit.io using this repo and `app.py` as the entrypoint.
3. In the app settings, set environment variables as needed:
   - `DRIVE_FILE_ID` — Google Drive file id for your `best.pt` (shareable link)
   - or `WEIGHTS_URL` — direct HTTP link to weights
   - or `WEIGHTS_PATH` — path inside the repo to the weights file

Notes:
- If your weights are large (>100 MB), prefer hosting them in Google Drive or a cloud bucket and use `DRIVE_FILE_ID` or `WEIGHTS_URL`.
- If using Google Drive, ensure the file is shared (Anyone with link) so Streamlit Cloud can download it.

