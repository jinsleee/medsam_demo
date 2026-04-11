"""
의료 샘플 이미지 생성 스크립트
실행: python generate_samples.py
"""
import numpy as np
from PIL import Image, ImageFilter
import os

OUT = os.path.join(os.path.dirname(__file__), 'MedSAM', 'assets', 'samples')
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)


def make_xray(size=512):
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    for dx in [-110, 110]:
        Y, X = np.ogrid[:size, :size]
        mask = ((X - (cx + dx)) / 90) ** 2 + ((Y - cy) / 130) ** 2 < 1
        img[mask] += 0.35
    Y, X = np.ogrid[:size, :size]
    mask = ((X - (cx + 20)) / 70) ** 2 + ((Y - (cy + 20)) / 80) ** 2 < 1
    img[mask] = np.maximum(img[mask], 0.55)
    img[cy + 120:cy + 145, cx - 200:cx + 200] = 0.6
    for i in range(7):
        y0 = cy - 100 + i * 30
        img[y0:y0 + 4, cx - 180:cx - 30] = 0.7
        img[y0:y0 + 4, cx + 30:cx + 180] = 0.7
    img += rng.normal(0, 0.03, img.shape)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert('RGB').filter(ImageFilter.GaussianBlur(1.2))


def make_brain_ct(size=512):
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    r = ((X - cx) ** 2 + (Y - cy) ** 2) ** 0.5
    img[(r > 180) & (r < 210)] = 0.9
    brain = r < 180
    img[brain] = 0.35 + rng.normal(0, 0.05, img.shape)[brain]
    for _ in range(12):
        bx = cx + rng.integers(-100, 100)
        by = cy + rng.integers(-100, 100)
        br = rng.integers(20, 55)
        bm = ((X - bx) ** 2 + (Y - by) ** 2) ** 0.5 < br
        img[bm & brain] += 0.12
    for offx in [-15, 15]:
        v = ((X - (cx + offx)) / 30) ** 2 + ((Y - cy) / 45) ** 2 < 1
        img[v & brain] = 0.1
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert('RGB').filter(ImageFilter.GaussianBlur(0.8))


def make_abdomen_ct(size=512):
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    r = ((X - cx) ** 2 + (Y - cy) ** 2) ** 0.5
    body = r < 220
    img[body] = 0.25 + rng.normal(0, 0.04, img.shape)[body]
    img[((X - cx) / 18) ** 2 + ((Y - cy) / 22) ** 2 < 1] = 1.0
    liver = (((X - (cx + 60)) / 100) ** 2 + ((Y - (cy - 20)) / 80) ** 2 < 1) & body
    img[liver] = 0.55 + rng.normal(0, 0.03, img.shape)[liver]
    spleen = (((X - (cx - 90)) / 45) ** 2 + ((Y - (cy - 10)) / 55) ** 2 < 1) & body
    img[spleen] = 0.5 + rng.normal(0, 0.03, img.shape)[spleen]
    for dx in [-130, 130]:
        kidney = (((X - (cx + dx)) / 30) ** 2 + ((Y - cy) / 40) ** 2 < 1) & body
        img[kidney] = 0.6
    tumor = ((X - (cx + 40)) / 22) ** 2 + ((Y - (cy - 30)) / 20) ** 2 < 1
    img[tumor & liver] = 0.78
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert('RGB').filter(ImageFilter.GaussianBlur(1.0))


def make_dermoscopy(size=512):
    base = rng.integers(160, 210, (size, size, 3), dtype=np.uint8)
    img = base.copy().astype(np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    lesion = ((X - cx) / 110) ** 2 + ((Y - cy) / 90) ** 2 < 1
    img[lesion] = [80, 45, 35]
    dark = ((X - (cx + 30)) / 50) ** 2 + ((Y - (cy - 20)) / 40) ** 2 < 1
    img[dark & lesion] = [40, 20, 15]
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.5))


if __name__ == '__main__':
    images = {
        'xray_chest.png':  make_xray(),
        'ct_brain.png':    make_brain_ct(),
        'ct_abdomen.png':  make_abdomen_ct(),
        'dermoscopy.png':  make_dermoscopy(),
    }
    for fname, img in images.items():
        path = os.path.join(OUT, fname)
        img.save(path)
        print(f'Saved: {path}')
    print('Done.')
