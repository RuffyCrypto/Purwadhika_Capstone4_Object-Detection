import streamlit as st
from PIL import Image
from pathlib import Path
import subprocess, sys, os, glob, shutil
from calorie_map import get_calorie_info

st.set_page_config(layout='wide')
st.title('FoodCal - YOLOv12 Inference (Patched)')

st.markdown('Upload an image. The app will call YOLOv12 detect script using the same Python interpreter and estimate calories based on detections.')

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader('Upload image', type=['jpg','jpeg','png'])
    conf = st.slider('Confidence threshold', 0.05, 0.9, 0.25)
    run_btn = st.button('Run Inference')

with col2:
    st.write('Instructions:')
    st.write('- Put your `best.pt` into `models/best.pt`')
    st.write('- If yolov12 not present, the app will clone it (requires internet during deploy/runtime)')

# Debug info in sidebar
with st.sidebar:
    st.header("Debug")
    try:
        import PIL as _PIL
        st.write("PIL version:", getattr(_PIL, '__version__', 'unknown'))
    except Exception as e:
        st.write("PIL import error:", e)
    st.write("Python executable:", sys.executable)
    st.write("Working dir:", os.getcwd())

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Input Image', use_container_width=True)

if run_btn:
    if uploaded is None:
        st.warning('Please upload an image first.')
    else:
        # save uploaded
        tmp_dir = Path('tmp_uploads')
        tmp_dir.mkdir(exist_ok=True)
        img_path = tmp_dir / uploaded.name
        img.save(img_path)

        # ensure yolov12 exists; clone if not
        if not Path('yolov12').exists():
            st.info('Cloning yolov12 repository (first-time).')
            proc = subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/sunsmarterjie/yolov12.git'], capture_output=True, text=True)
            # fallback to git clone if pip install as git fails
            if proc.returncode != 0:
                st.text(proc.stdout)
                st.text(proc.stderr)
                st.info('Trying git clone...')
                git_proc = subprocess.run(['git', 'clone', 'https://github.com/sunsmarterjie/yolov12.git'], capture_output=True, text=True)
                st.text(git_proc.stdout)
                if git_proc.returncode != 0:
                    st.error('Failed cloning yolov12. See logs above.')
                    st.stop()

        # check model file
        model_path = Path('models') / 'best.pt'
        if not model_path.exists():
            st.error('Model file models/best.pt not found. Please place your best.pt into models/ folder.')
            st.stop()

        # build command using same interpreter
        python_exec = sys.executable
        detect_script = os.path.join('yolov12', 'detect.py')
        # use absolute paths to avoid cwd confusion
        cmd = [
            python_exec,
            detect_script,
            '--weights', str(model_path.resolve()),
            '--source', str(img_path.resolve()),
            '--conf', str(conf),
            '--save-txt',
            '--save-img'
        ]

        # if yolov12 expects to be run from its own directory, set cwd accordingly
        cwd = 'yolov12' if Path('yolov12').exists() else os.getcwd()
        st.info('Running detection...')
        st.text('Command: ' + ' '.join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=os.environ.copy())
        st.text('STDOUT:\n' + (proc.stdout or ''))
        st.text('STDERR:\n' + (proc.stderr or ''))

        if proc.returncode != 0:
            st.error('Detection failed (returncode != 0). See stderr above.')
            st.stop()

        # find latest runs/detect/exp* under yolov12 (or cwd if detect placed runs elsewhere)
        runs = sorted(glob.glob(os.path.join(cwd, 'runs', 'detect', 'exp*')))
        if not runs:
            st.error('No detection run folder found under yolov12/runs/detect/')
            st.stop()
        latest = runs[-1]
        st.success(f'Detection completed. Output folder: {latest}')

        # show annotated image (same filename under latest)
        out_img = os.path.join(latest, os.path.basename(str(img_path)))
        if os.path.exists(out_img):
            st.image(out_img, caption='Annotated', use_container_width=True)
        else:
            st.warning('Annotated image not found at ' + out_img)

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
