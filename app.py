from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json, zipfile, io, base64, uuid, shutil
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store: { image_id: { filename, path, width, height, labels: [...] } }
image_store = {}

# Lazy-load YOLO so startup is fast
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')  # downloads ~6 MB on first run
    return _model

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_images():
    files = request.files.getlist('images')
    results = []
    for f in files:
        img_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(f.filename)[1].lower()
        save_path = os.path.join(UPLOAD_FOLDER, f'{img_id}{ext}')
        f.save(save_path)
        img = Image.open(save_path).convert('RGB')
        w, h = img.size
        image_store[img_id] = {
            'filename': f.filename,
            'path': save_path,
            'width': w,
            'height': h,
            'labels': []
        }
        results.append({'id': img_id, 'filename': f.filename, 'width': w, 'height': h})
    return jsonify({'uploaded': results})

@app.route('/api/image/<img_id>')
def serve_image(img_id):
    if img_id not in image_store:
        return jsonify({'error': 'not found'}), 404
    path = image_store[img_id]['path']
    img = Image.open(path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': f'data:image/jpeg;base64,{b64}',
                    'width': image_store[img_id]['width'],
                    'height': image_store[img_id]['height']})

@app.route('/api/autolabel/<img_id>', methods=['POST'])
def auto_label(img_id):
    if img_id not in image_store:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    conf = float(data.get('confidence', 0.35))
    path = image_store[img_id]['path']
    model = get_model()
    results = model(path, conf=conf, verbose=False)[0]
    labels = []
    iw, ih = image_store[img_id]['width'], image_store[img_id]['height']
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        label = results.names[cls_id]
        labels.append({
            'id': str(uuid.uuid4())[:8],
            'label': label,
            'x': round(x1 / iw, 6),
            'y': round(y1 / ih, 6),
            'w': round((x2 - x1) / iw, 6),
            'h': round((y2 - y1) / ih, 6),
            'confidence': round(conf_val, 3),
            'auto': True
        })
    image_store[img_id]['labels'] = labels
    return jsonify({'labels': labels, 'count': len(labels)})

@app.route('/api/labels/<img_id>', methods=['GET'])
def get_labels(img_id):
    if img_id not in image_store:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'labels': image_store[img_id]['labels']})

@app.route('/api/labels/<img_id>', methods=['POST'])
def save_labels(img_id):
    if img_id not in image_store:
        return jsonify({'error': 'not found'}), 404
    data = request.json
    image_store[img_id]['labels'] = data.get('labels', [])
    return jsonify({'saved': True, 'count': len(image_store[img_id]['labels'])})

@app.route('/api/export', methods=['POST'])
def export_dataset():
    data = request.json or {}
    fmt = data.get('format', 'yolo')          # yolo | coco | csv
    ids = data.get('ids') or list(image_store.keys())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:

        if fmt == 'yolo':
            # YOLO: images/ + labels/ + classes.txt
            all_labels = set()
            for img_id in ids:
                for lb in image_store[img_id]['labels']:
                    all_labels.add(lb['label'])
            class_list = sorted(all_labels)
            zf.writestr('classes.txt', '\n'.join(class_list))

            for img_id in ids:
                entry = image_store[img_id]
                # copy image
                img_data = open(entry['path'], 'rb').read()
                img_name = f"{img_id}{os.path.splitext(entry['path'])[1]}"
                zf.writestr(f'images/{img_name}', img_data)
                # write label file
                lines = []
                for lb in entry['labels']:
                    cls_idx = class_list.index(lb['label'])
                    cx = lb['x'] + lb['w'] / 2
                    cy = lb['y'] + lb['h'] / 2
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {lb['w']:.6f} {lb['h']:.6f}")
                zf.writestr(f'labels/{img_id}.txt', '\n'.join(lines))
            zf.writestr('data.yaml',
                f"train: images/\nval: images/\nnc: {len(class_list)}\nnames: {class_list}\n")

        elif fmt == 'coco':
            coco = {'images': [], 'annotations': [], 'categories': []}
            cat_map = {}
            ann_id = 1
            for img_idx, img_id in enumerate(ids):
                entry = image_store[img_id]
                coco['images'].append({'id': img_idx, 'file_name': entry['filename'],
                                       'width': entry['width'], 'height': entry['height']})
                for lb in entry['labels']:
                    if lb['label'] not in cat_map:
                        cat_map[lb['label']] = len(cat_map) + 1
                        coco['categories'].append({'id': cat_map[lb['label']], 'name': lb['label']})
                    x_abs = lb['x'] * entry['width']
                    y_abs = lb['y'] * entry['height']
                    w_abs = lb['w'] * entry['width']
                    h_abs = lb['h'] * entry['height']
                    coco['annotations'].append({
                        'id': ann_id, 'image_id': img_idx,
                        'category_id': cat_map[lb['label']],
                        'bbox': [round(x_abs,2), round(y_abs,2), round(w_abs,2), round(h_abs,2)],
                        'area': round(w_abs * h_abs, 2), 'iscrowd': 0
                    })
                    ann_id += 1
                img_data = open(entry['path'], 'rb').read()
                img_name = f"{img_id}{os.path.splitext(entry['path'])[1]}"
                zf.writestr(f'images/{img_name}', img_data)
            zf.writestr('annotations.json', json.dumps(coco, indent=2))

        elif fmt == 'csv':
            rows = ['image_id,filename,label,x,y,w,h,confidence,auto']
            for img_id in ids:
                entry = image_store[img_id]
                for lb in entry['labels']:
                    rows.append(f"{img_id},{entry['filename']},{lb['label']},"
                                f"{lb['x']},{lb['y']},{lb['w']},{lb['h']},"
                                f"{lb.get('confidence','')},{lb.get('auto',False)}")
            zf.writestr('labels.csv', '\n'.join(rows))

    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name=f'dataset_{fmt}.zip')

@app.route('/api/stats')
def stats():
    total_labels = sum(len(v['labels']) for v in image_store.values())
    all_classes = {}
    for v in image_store.values():
        for lb in v['labels']:
            all_classes[lb['label']] = all_classes.get(lb['label'], 0) + 1
    return jsonify({
        'total_images': len(image_store),
        'total_labels': total_labels,
        'classes': all_classes
    })

@app.route('/api/clear', methods=['POST'])
def clear_all():
    image_store.clear()
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return jsonify({'cleared': True})

if __name__ == '__main__':
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   AutoLabel — running on :5000       ║")
    print("  ║   Open http://localhost:5000          ║")
    print("  ╚══════════════════════════════════════╝\n")
    if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)