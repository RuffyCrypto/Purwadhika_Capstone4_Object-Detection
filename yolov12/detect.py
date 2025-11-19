#!/usr/bin/env python3
import argparse
import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

def next_exp_dir(base='runs/detect'):
    Path(base).mkdir(parents=True, exist_ok=True)
    exps = sorted(glob.glob(os.path.join(base, 'exp*')))
    if not exps:
        return os.path.join(base, 'exp1')
    last = exps[-1]
    try:
        n = int(Path(last).name.replace('exp', '')) + 1
    except Exception:
        n = len(exps) + 1
    return os.path.join(base, f'exp{n}')

def xyxy_to_yolo(xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h

def draw_boxes_pil(img, boxes, scores, classes, names, conf_thres):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        if s < conf_thres:
            continue
        label = f"{names[int(c)]} {s:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        if font:
            draw.text((x1 + 3, y1 + 3), label, fill='red', font=font)
    return img

def run(weights, source, conf, save_txt, save_img):
    # Prefer ultralytics YOLO API if available (avoids torch.hub cache issues)
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        use_ultralytics = True
    except Exception:
        # fallback to torch.hub (yolov5 repo)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True, trust_repo=True)
        model.conf = conf
        use_ultralytics = False

    # prepare output folder
    base_runs = os.path.join(Path(__file__).parent, 'runs', 'detect')
    out_dir = next_exp_dir(base=base_runs)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    labels_dir = os.path.join(out_dir, 'labels')
    if save_txt:
        Path(labels_dir).mkdir(parents=True, exist_ok=True)

    # collect sources
    srcs = []
    p = Path(source)
    if p.is_dir():
        srcs = sorted([str(x) for x in p.glob('*') if x.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    elif p.is_file():
        srcs = [str(p)]
    else:
        srcs = sorted(glob.glob(source))
        if not srcs:
            raise FileNotFoundError(f"No source files found for: {source}")

    print(f"Found {len(srcs)} images. Saving results to: {out_dir}")

    for img_path in srcs:
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # inference (handle ultralytics YOLO vs torch.hub result formats)
        results = model(img)
        if use_ultralytics:
            r = results[0]
            if hasattr(r, 'boxes') and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
            else:
                boxes = np.zeros((0,4)); scores = np.zeros((0,)); classes = np.zeros((0,))
        else:
            preds = results.pred[0]  # tensor Nx6 (x1,y1,x2,y2,conf,cls)
            preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)
            boxes = preds[:, :4] if preds.size else np.zeros((0, 4))
            scores = preds[:, 4] if preds.size else np.zeros((0,))
            classes = preds[:, 5] if preds.size else np.zeros((0,))

        stem = Path(img_path).stem
        out_img_path = os.path.join(out_dir, Path(img_path).name)
        label_txt_path = os.path.join(labels_dir, stem + '.txt')

        # save txt in YOLO format
        if save_txt:
            with open(label_txt_path, 'w') as f:
                for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
                    if s < conf:
                        continue
                    cx, cy, w, h = xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
                    f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # save annotated image
        if save_img:
            annotated = draw_boxes_pil(img.copy(), boxes, scores, classes, model.names, conf)
            annotated.save(out_img_path)

        print(f"Processed {img_path} -> {out_img_path if save_img else out_dir}")

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description="Simple detect wrapper (writes runs/detect/exp*/ images + labels)"
    )
    parser.add_argument('--weights', type=str, default='models/best.pt', help='path to .pt weights')
    parser.add_argument('--source', type=str, required=True, help='image file, dir, or glob pattern')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold (0-1)')
    parser.add_argument('--save-txt', action='store_true', help='save labels in YOLO format')
    parser.add_argument('--save-img', action='store_true', help='save annotated images')
    args = parser.parse_args()
    run(args.weights, args.source, args.conf, args.save_txt, args.save_img)

if __name__ == "__main__":
    parse_args_and_run()
