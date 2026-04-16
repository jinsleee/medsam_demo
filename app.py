"""
SAM vs MedSAM Comparison Demo
- 좌측: 원본 SAM (일반 이미지 학습)
- 우측: MedSAM (의료영상 파인튜닝, bowang-lab)

논문 발표용: Medical SAM Adapter (Wu et al., 2025)가 지적하는
'SAM의 의료영상 한계'와 'MedSAM 같은 파인튜닝 방식'의 차이를
라이브로 비교합니다.
"""
import os
import sys
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MedSAM'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

BASE_DIR = os.path.dirname(__file__)
SAM_CKPT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'sam_vit_b.pth')
MEDSAM_CKPT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'medsam_vit_b.pth')
SAMPLE_DIR = os.path.join(BASE_DIR, 'MedSAM', 'assets')
SAMPLE_MEDICAL_DIR = os.path.join(BASE_DIR, 'MedSAM', 'assets', 'samples')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
IMAGE_DIR = os.path.join(BASE_DIR, 'image')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# 두 개의 모델을 각각 로드
sam_model = None
medsam_model = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_single_model(ckpt_path, tag):
    """SAM 아키텍처는 동일, 가중치만 다름"""
    if not os.path.exists(ckpt_path):
        print(f"[{tag}] checkpoint not found: {ckpt_path}")
        return None
    try:
        from segment_anything import sam_model_registry
        model = sam_model_registry['vit_b'](checkpoint=ckpt_path)
        model.to(DEVICE)
        model.eval()
        print(f"[{tag}] loaded on {DEVICE}")
        return model
    except Exception as e:
        print(f"[{tag}] load error: {e}")
        return None


def load_models():
    """SAM과 MedSAM을 둘 다 로드. 하나만 있어도 그건 작동시킴"""
    global sam_model, medsam_model
    if sam_model is None:
        sam_model = _load_single_model(SAM_CKPT_PATH, 'SAM')
    if medsam_model is None:
        medsam_model = _load_single_model(MEDSAM_CKPT_PATH, 'MedSAM')
    return sam_model is not None, medsam_model is not None


# ---------- SAM 추론 (원본 방식) ----------
def run_sam(image_np, box):
    """원본 SAM: 이미지를 그대로 넣고 박스 프롬프트"""
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam_model)

    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    image_np = image_np.astype(np.uint8)

    predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
    y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))

    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=np.array([[x1, y1, x2, y2]]),
        multimask_output=False
    )
    return masks[0], float(scores[0])


# ---------- MedSAM 추론 (공식 전처리 방식) ----------
def run_medsam(image_np, box):
    """
    MedSAM 공식 추론 파이프라인 (bowang-lab/MedSAM):
    1. 이미지를 1024x1024로 리사이즈
    2. [0, 1]로 정규화
    3. 박스 좌표도 1024 스케일로 변환
    4. image encoder로 임베딩 추출 후 mask decoder 호출
    """
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    image_np = image_np.astype(np.uint8)

    H, W = image_np.shape[:2]

    # 1. 1024x1024로 리사이즈
    img_pil = Image.fromarray(image_np).resize((1024, 1024), Image.BILINEAR)
    img_1024 = np.array(img_pil)

    # 2. 정규화
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 3. 박스 좌표도 1024 스케일로
    x1, y1, x2, y2 = box
    box_1024 = np.array([
        x1 * 1024.0 / W, y1 * 1024.0 / H,
        x2 * 1024.0 / W, y2 * 1024.0 / H
    ])[None, :]  # (1, 4)

    with torch.no_grad():
        # Image embedding
        image_embedding = medsam_model.image_encoder(img_tensor)

        # Prompt encoder
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=DEVICE)
        box_torch = box_torch[:, None, :]  # (1, 1, 4)
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

        # Mask decoder
        low_res_logits, iou_pred = medsam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # 원본 해상도로 업샘플
        low_res_pred = F.interpolate(
            low_res_logits, size=(H, W), mode='bilinear', align_corners=False
        )
        low_res_pred = torch.sigmoid(low_res_pred).squeeze().cpu().numpy()
        mask = (low_res_pred > 0.5).astype(bool)

    return mask, float(iou_pred.squeeze().cpu().numpy())


# ---------- 시각화 & 유틸 ----------
def mask_to_overlay(image_np, mask, color=(0, 255, 100), alpha=0.45):
    overlay = image_np.copy().astype(np.float32)
    for c, val in enumerate(color):
        overlay[:, :, c] = np.where(
            mask, overlay[:, :, c] * (1 - alpha) + val * alpha, overlay[:, :, c]
        )
    import cv2
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay.astype(np.uint8)


def np_to_base64(img_np):
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def compute_dice(mask_a, mask_b):
    """두 마스크의 Dice 유사도"""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return float(2.0 * inter / total)


# ---------- Flask 엔드포인트 ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    return jsonify({
        'sam_checkpoint_exists': os.path.exists(SAM_CKPT_PATH),
        'medsam_checkpoint_exists': os.path.exists(MEDSAM_CKPT_PATH),
        'sam_loaded': sam_model is not None,
        'medsam_loaded': medsam_model is not None,
        'device': str(DEVICE),
    })


@app.route('/load_model', methods=['POST'])
def api_load_model():
    sam_ok, medsam_ok = load_models()
    return jsonify({
        'sam_loaded': sam_ok,
        'medsam_loaded': medsam_ok,
        'success': sam_ok or medsam_ok,
    })


@app.route('/medical_samples')
def medical_samples():
    meta = {
        'xray_chest.png': {'label': '흉부 X-ray', 'icon': '🫁',
                           'hint': '폐·심장 영역을 박스로 감싸보세요',
                           'box': [130, 80, 380, 420]},
        'ct_brain.png': {'label': '뇌 CT', 'icon': '🧠',
                         'hint': '뇌실 또는 특정 뇌 영역을 선택',
                         'box': [150, 130, 360, 380]},
        'ct_abdomen.png': {'label': '복부 CT', 'icon': '🫀',
                           'hint': '간·신장·종양 부위를 드래그',
                           'box': [230, 150, 430, 340]},
        'dermoscopy.png': {'label': '피부 병변', 'icon': '🔬',
                           'hint': '병변(어두운 부위) 주변을 박스로',
                           'box': [130, 140, 380, 370]},
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


@app.route('/list_images')
def list_images():
    ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    files = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        if os.path.splitext(f)[1].lower() in ALLOWED:
            files.append(f)
    return jsonify({'images': files})


@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route('/segment', methods=['POST'])
def segment():
    """
    핵심 변경점: SAM과 MedSAM 결과를 둘 다 반환
    둘 중 하나만 로드되어 있으면 그 쪽 결과만 반환 (나머진 null)
    """
    if sam_model is None and medsam_model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400

    try:
        data = request.get_json()
        image_b64 = data.get('image')
        box = data.get('box')
        if not image_b64 or not box:
            return jsonify({'error': '이미지와 박스가 필요합니다.'}), 400

        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_np = np.array(image)

        results = {'sam': None, 'medsam': None, 'comparison': None}

        # SAM 실행
        sam_mask = None
        if sam_model is not None:
            sam_mask, sam_score = run_sam(image_np, box)
            results['sam'] = {
                'overlay': np_to_base64(mask_to_overlay(image_np, sam_mask,
                                                       color=(255, 100, 100))),
                'score': round(sam_score, 4),
                'area_px': int(sam_mask.sum()),
                'area_pct': round(sam_mask.sum() / sam_mask.size * 100, 2),
            }

        # MedSAM 실행
        medsam_mask = None
        if medsam_model is not None:
            medsam_mask, medsam_score = run_medsam(image_np, box)
            results['medsam'] = {
                'overlay': np_to_base64(mask_to_overlay(image_np, medsam_mask,
                                                       color=(0, 255, 100))),
                'score': round(medsam_score, 4),
                'area_px': int(medsam_mask.sum()),
                'area_pct': round(medsam_mask.sum() / medsam_mask.size * 100, 2),
            }

        # 두 결과 비교 지표
        if sam_mask is not None and medsam_mask is not None:
            dice = compute_dice(sam_mask, medsam_mask)
            area_diff_px = int(abs(sam_mask.sum() - medsam_mask.sum()))
            results['comparison'] = {
                'dice_between': round(dice, 4),
                'area_diff_px': area_diff_px,
                'area_diff_pct': round(area_diff_px / sam_mask.size * 100, 2),
            }

        return jsonify(results)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=== SAM vs MedSAM Demo ===")
    print(f"SAM checkpoint:    {SAM_CKPT_PATH} (exists: {os.path.exists(SAM_CKPT_PATH)})")
    print(f"MedSAM checkpoint: {MEDSAM_CKPT_PATH} (exists: {os.path.exists(MEDSAM_CKPT_PATH)})")
    print(f"Device: {DEVICE}")
    app.run(debug=True, host='0.0.0.0', port=5020)