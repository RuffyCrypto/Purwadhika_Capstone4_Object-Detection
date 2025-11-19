# app/streamlit_app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import subprocess, os, glob
from calorie_map import get_calorie_info
import sys, os

st.set_page_config(layout='wide')
st.title('FoodCal - YOLOv12 Inference')

st.markdown('Upload an image. The app will call YOLOv12 detect script and estimate calories based on detections.')

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader('Upload image', type=['jpg','jpeg','png'])
    conf = st.slider('Confidence threshold', 0.05, 0.9, 0.25)
    run_btn = st.button('Run Inference')

with col2:
    st.write('Instructions:')
    st.write('- Put your `best.pt` into `models/best.pt`')
    st.write('- If repo not cloned, run `bash scripts/clone_yolov12.sh` from project root')

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Input Image', use_column_width=True)

if run_btn:
    if uploaded is None:
        st.warning('Please upload an image first.')
    else:
        # save uploaded
        tmp_dir = Path('tmp_uploads')
        tmp_dir.mkdir(exist_ok=True)
        img_path = tmp_dir / uploaded.name
        img.save(img_path)

        # ensure yolov12 exists
        if not Path('yolov12').exists():
            st.info('Cloning yolov12 repository (first-time)...')
            proc = subprocess.run(['bash','scripts/clone_yolov12.sh'], capture_output=True, text=True)
            st.text(proc.stdout)
            if proc.returncode != 0:
                st.error('Failed cloning yolov12. See logs.')
                st.text(proc.stderr)
                st.stop()

        # check model
        model_path = Path('models') / 'best.pt'
        if not model_path.exists():
            st.error('Model file models/best.pt not found. Please place your best.pt into models/ folder.')
            st.stop()

        # call yolov12 detect
        # Use the same Python interpreter that's running Streamlit to run the detect script,
# so the subprocess uses the same environment (packages like Pillow will be available).
python_exec = sys.executable  # ensures same interpreter/environment as Streamlit
detect_script = os.path.join('yolov12', 'detect.py')
cmd = [
    python_exec,
    detect_script,
    '--weights', str(model_path),
    '--source', str(img_path),
    '--conf', str(conf),
    '--save-txt',
    '--save-img'
]
env = os.environ.copy()
# If detect.py expects to run from the yolov12 repo, set cwd accordingly; otherwise use project root.
cwd = os.path.join(os.getcwd(), 'yolov12') if os.path.isdir('yolov12') else os.getcwd()
# Run subprocess and capture output
proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
# Show output for debugging
st.text(proc.stdout)
if proc.returncode != 0:
    st.error('Detection failed (returncode != 0). See stderr below:')
    st.text(proc.stderr)
    st.stop()

        st.text(proc.stdout)
        if proc.returncode != 0:
            st.error('Detection failed. See stderr:')
            st.text(proc.stderr)
            st.stop()

        # find latest runs/detect/exp*
        runs = sorted(glob.glob('yolov12/runs/detect/exp*'))
        if not runs:
            st.error('No detection run folder found under yolov12/runs/detect/')
            st.stop()
        latest = runs[-1]
        st.success(f'Detection completed. Output folder: {latest}')

        # show annotated image (same filename)
        out_img = os.path.join(latest, os.path.basename(str(img_path)))
        if os.path.exists(out_img):
            st.image(out_img, caption='Annotated', use_column_width=True)
        else:
            st.warning('Annotated image not found.')

        # read label txt
        label_txt = os.path.join(latest, 'labels', Path(img_path).stem + '.txt')
        counts = {}
        total_cal = 0
        if os.path.exists(label_txt):
            with open(label_txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    counts[cls] = counts.get(cls, 0) + 1
                    info = get_calorie_info(cls)
                    total_cal += info.get('cal', 0)
        else:
            st.warning('Label txt not found; ensure yolov12 detect used --save-txt')

        st.subheader('Detections')
        if counts:
            for cls, cnt in counts.items():
                info = get_calorie_info(cls)
                st.write(f"{info['label']} (class {cls}) : {cnt} → {info['cal']} per unit")
            st.metric('Estimated Total Calories', f"{total_cal} kcal")
        else:
            st.write('No objects counted.')
