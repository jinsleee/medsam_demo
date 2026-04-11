import os
import sys
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MedSAM'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'medsam_vit_b.pth')
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'MedSAM', 'assets')
SAMPLE_MEDICAL_DIR = os.path.join(os.path.dirname(__file__), 'MedSAM', 'assets', 'samples')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

medsam_model = None

def load_model():
    global medsam_model
    if medsam_model is not None:
        return True
    if not os.path.exists(CHECKPOINT_PATH):
        return False
    try:
        from segment_anything import sam_model_registry
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry['vit_b'](checkpoint=CHECKPOINT_PATH)
        sam.to(device)
        sam.eval()
        medsam_model = sam
        print(f"Model loaded on {device}")
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False


def run_medsam(image_np, box):
    from segment_anything import SamPredictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(medsam_model)

    # Ensure RGB
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    image_np = image_np.astype(np.uint8)
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    input_box = np.array([[x1, y1, x2, y2]])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False
    )
    return masks[0], float(scores[0])


def mask_to_overlay(image_np, mask, color=(0, 255, 100), alpha=0.45):
    overlay = image_np.copy().astype(np.float32)
    for c, val in enumerate(color):
        overlay[:, :, c] = np.where(mask, overlay[:, :, c] * (1 - alpha) + val * alpha, overlay[:, :, c])
    # Draw contour border
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay.astype(np.uint8)


def np_to_base64(img_np):
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    model_ready = os.path.exists(CHECKPOINT_PATH)
    loaded = medsam_model is not None
    checkpoint_size = 0
    if model_ready:
        checkpoint_size = os.path.getsize(CHECKPOINT_PATH) // (1024 * 1024)
    return jsonify({
        'checkpoint_exists': model_ready,
        'model_loaded': loaded,
        'checkpoint_size_mb': checkpoint_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/load_model', methods=['POST'])
def api_load_model():
    success = load_model()
    return jsonify({'success': success, 'loaded': medsam_model is not None})


@app.route('/samples')
def samples():
    files = []
    for fname in os.listdir(SAMPLE_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(fname)
    return jsonify({'samples': files})


@app.route('/sample_image/<filename>')
def sample_image(filename):
    return send_from_directory(SAMPLE_DIR, filename)


@app.route('/medical_samples')
def medical_samples():
    meta = {
        'xray_chest.png':   {'label': '흉부 X-ray', 'icon': '🫁', 'hint': '폐·심장 영역을 박스로 감싸보세요', 'box': [130, 80, 380, 420]},
        'ct_brain.png':     {'label': '뇌 CT',      'icon': '🧠', 'hint': '뇌실 또는 특정 뇌 영역을 선택하세요', 'box': [150, 130, 360, 380]},
        'ct_abdomen.png':   {'label': '복부 CT',    'icon': '🫀', 'hint': '간·신장·종양 부위를 드래그하세요', 'box': [230, 150, 430, 340]},
        'dermoscopy.png':   {'label': '피부 병변',  'icon': '🔬', 'hint': '병변(어두운 부위) 주변을 박스로 선택하세요', 'box': [130, 140, 380, 370]},
    }
    result = []
    for fname, info in meta.items():
        path = os.path.join(SAMPLE_MEDICAL_DIR, fname)
        if os.path.exists(path):
            result.append({'filename': fname, **info})
    return jsonify({'samples': result})


@app.route('/medical_image/<filename>')
def medical_image(filename):
    return send_from_directory(SAMPLE_MEDICAL_DIR, filename)


@app.route('/segment', methods=['POST'])
def segment():
    if medsam_model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.'}), 400

    try:
        data = request.get_json()
        image_b64 = data.get('image')
        box = data.get('box')  # [x1, y1, x2, y2] in original image coords

        if not image_b64 or not box:
            return jsonify({'error': '이미지와 박스 좌표가 필요합니다.'}), 400

        # Decode image
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_np = np.array(image)

        mask, score = run_medsam(image_np, box)

        overlay_np = mask_to_overlay(image_np, mask)
        overlay_b64 = np_to_base64(overlay_np)

        # Mask stats
        area_px = int(mask.sum())
        total_px = mask.size
        area_pct = round(area_px / total_px * 100, 2)

        return jsonify({
            'overlay': overlay_b64,
            'score': round(score, 4),
            'area_px': area_px,
            'area_pct': area_pct,
            'mask_size': list(mask.shape)
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=== MedSAM Web UI ===")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint exists: {os.path.exists(CHECKPOINT_PATH)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
