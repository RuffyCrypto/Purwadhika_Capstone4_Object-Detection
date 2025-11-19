import streamlit as st
from PIL import Image
from pathlib import Path
import subprocess, sys, os, glob
from calorie_map import get_calorie_info

st.set_page_config(layout='wide')
st.title('FoodCal - YOLOv12 Inference (Stable)')

st.markdown('Upload an image. The app will call YOLOv12 detect script using the same Python interpreter and estimate calories based on detections.')

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader('Upload image', type=['jpg','jpeg','png'])
    conf = st.slider('Confidence threshold', 0.05, 0.9, 0.25)
    run_btn = st.button('Run Inference')

with col2:
    st.write('Instructions:')
    st.write('- Put your `best.pt` into `models/best.pt`')
    st.write('- The app will try to clone yolov12 repo if not present (requires internet during runtime)')

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

def _safe_text(s):
    if s is None:
        return ""
    if isinstance(s, bytes):
        try:
            return s.decode('utf-8', errors='replace')
        except Exception:
            return str(s)
    return str(s)

if run_btn:
    if uploaded is None:
        st.warning('Please upload an image first.')
    else:
        # save uploaded locally
        tmp_dir = Path('tmp_uploads')
        tmp_dir.mkdir(exist_ok=True)
        img_path = tmp_dir / uploaded.name
        img.save(img_path)

        # ensure yolov12 exists; try pip install git+ then fallback to git clone
        if not Path('yolov12').exists():
            st.info('Installing or cloning yolov12 repository (first-time).')
            proc_install = subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/sunsmarterjie/yolov12.git'], capture_output=True, text=True)
            st.text('pip install stdout:\n' + _safe_text(proc_install.stdout))
            st.text('pip install stderr:\n' + _safe_text(proc_install.stderr))
            if proc_install.returncode != 0:
                st.info('pip install failed, trying git clone...')
                proc_clone = subprocess.run(['git', 'clone', 'https://github.com/sunsmarterjie/yolov12.git'], capture_output=True, text=True)
                st.text('git clone stdout:\n' + _safe_text(proc_clone.stdout))
                st.text('git clone stderr:\n' + _safe_text(proc_clone.stderr))
                if proc_clone.returncode != 0:
                    st.error('Failed to get yolov12 repo. Cannot proceed.')
                    st.stop()

        # check model existence
        model_path = Path('models') / 'best.pt'
        if not model_path.exists():
            st.error('Model file models/best.pt not found. Please place your best.pt into models/ folder.')
            st.stop()

        # prepare detect command using same python interpreter
        python_exec = sys.executable
        cwd = 'yolov12' if Path('yolov12').exists() else os.getcwd()
        if cwd == 'yolov12':
            detect_script = 'detect.py'
        else:
            detect_script = os.path.join('yolov12', 'detect.py')

        cmd = [
            python_exec,
            detect_script,
            '--weights', str(model_path.resolve()),
            '--source', str(img_path.resolve()),
            '--conf', str(conf),
            '--save-txt',
            '--save-img'
        ]

        st.info('Running detection...')
        st.text('Working dir: ' + str(cwd))
        st.text('Command: ' + ' '.join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=False, cwd=cwd, env=os.environ.copy())
        # proc.stdout/proc.stderr might be bytes; decode safely
        stdout = proc.stdout.decode('utf-8', errors='replace') if isinstance(proc.stdout, (bytes, bytearray)) else str(proc.stdout)
        stderr = proc.stderr.decode('utf-8', errors='replace') if isinstance(proc.stderr, (bytes, bytearray)) else str(proc.stderr)
        st.text('STDOUT:\n' + stdout)
        st.text('STDERR:\n' + stderr)

        if proc.returncode != 0:
            st.error('Detection failed (returncode != 0). See stderr above.')
            st.stop()

        # find latest runs/detect/exp*
        runs = sorted(glob.glob(os.path.join(cwd, 'runs', 'detect', 'exp*')))
        if not runs:
            st.error('No detection run folder found under yolov12/runs/detect/')
            st.stop()
        latest = runs[-1]
        st.success(f'Detection completed. Output folder: {latest}')

        # show annotated image
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
