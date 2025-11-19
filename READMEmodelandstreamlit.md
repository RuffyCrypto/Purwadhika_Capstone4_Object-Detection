# FoodCal - YOLOv12 Inference Package (Streamlit)

This package helps you run inference with a YOLOv12 `best.pt` model and a Streamlit app
that estimates calories based on detected food items.

## Steps
1. Place your `best.pt` into the `models/` folder (path: `models/best.pt`).
2. Install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Clone YOLOv12 repo (the script will do it automatically if not present):
   ```
   bash scripts/clone_yolov12.sh
   ```
4. Run Streamlit:
   ```
   streamlit run app/streamlit_app.py
   ```

## Notes
- The Streamlit app will call the YOLOv12 `detect.py` script using your `models/best.pt`.
- After inference, the app reads the generated label `.txt` in YOLO format and counts detected classes.
- Edit `src/calorie_map.py` or provide your `data.yaml` to map class indexes to calorie values.
