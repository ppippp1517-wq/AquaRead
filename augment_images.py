import os
import cv2
import random
import numpy as np
from pathlib import Path

# -------------------- CONFIG --------------------
INPUT_DIR    = r"D:\projectCPE\dataset\images\capture_images"
OUTPUT_DIR   = r"D:\projectCPE\dataset\images\augmented_images"
AUG_PER_IMG  = 10
OUT_SIZE     = None
SEED         = 2025

# ปรับช่วงให้เหมาะกับงานตัวเลข (ไม่มี rotation/flip แล้ว)
SCALE_RANGE    = (0.95, 1.05)   # ซูม ±5% (คงความสมจริง)
SHIFT_FRAC     = (-0.03, 0.03)  # เลื่อนตามสัดส่วนภาพ ±3%
GAMMA_RANGE    = (0.90, 1.10)
BRIGHT_DELTA   = (-15, 15)
CONTRAST_ALPHA = (0.95, 1.10)
CROP_FRAC      = (0.00, 0.05)
BLUR_K         = [0, 1, 3]
NOISE_SIGMA    = (0.0, 6.0)
JPEG_QUALITY   = (70, 95)

# -------------------- UTILS --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_bgr3(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def random_scale_shift(img):
    """แปลง affine แบบไม่มีการหมุน (rotation=0) มีแค่ scale + translation"""
    h, w = img.shape[:2]
    scale = random.uniform(*SCALE_RANGE)
    tx = random.uniform(*SHIFT_FRAC) * w
    ty = random.uniform(*SHIFT_FRAC) * h
    # rotation=0 → cos=1, sin=0
    M = np.array([[scale, 0.0, tx],
                  [0.0,   scale, ty]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def random_crop_resize(img):
    h, w = img.shape[:2]
    frac = random.uniform(*CROP_FRAC)
    if frac <= 0:
        return img
    dx = int(frac * w)
    dy = int(frac * h)
    x1 = random.randint(0, dx)
    y1 = random.randint(0, dy)
    x2 = w - random.randint(0, dx)
    y2 = h - random.randint(0, dy)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return img
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def adjust_brightness_contrast_gamma(img):
    alpha = random.uniform(*CONTRAST_ALPHA)
    beta = random.uniform(*BRIGHT_DELTA)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    gamma = random.uniform(*GAMMA_RANGE)
    if abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, table)
    return out

def random_blur(img):
    k = random.choice(BLUR_K)
    if k and k % 2 == 1:
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

def add_gaussian_noise(img):
    sigma = random.uniform(*NOISE_SIGMA)
    if sigma <= 0.01:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def jpeg_compress_artifact(img):
    q = int(random.uniform(*JPEG_QUALITY))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode('.jpg', img, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img

def pipeline(img):
    out = img.copy()
    out = random_scale_shift(out)          # ไม่มี rotation
    out = random_crop_resize(out)
    # ไม่มี flip
    out = adjust_brightness_contrast_gamma(out)
    out = random_blur(out)
    out = add_gaussian_noise(out)
    out = jpeg_compress_artifact(out)
    return out

def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    img = to_bgr3(img)
    return img

def write_image(path, img):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

# -------------------- MAIN --------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    input_paths = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    for ext in exts:
        input_paths.extend(Path(INPUT_DIR).rglob(ext))
    input_paths = sorted(input_paths)

    if not input_paths:
        print(f"❌ ไม่พบภาพใน {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(input_paths)} images")
    counter = 0  # ตั้งชื่อ img0, img1, ...
    for idx, src_path in enumerate(input_paths, 1):
        try:
            img = read_image(src_path)
            if OUT_SIZE is not None:
                if not (isinstance(OUT_SIZE, (tuple, list)) and len(OUT_SIZE) == 2):
                    raise ValueError("OUT_SIZE ต้องเป็น (width, height) หรือ None")
                img_base = cv2.resize(img, OUT_SIZE, interpolation=cv2.INTER_AREA)
            else:
                img_base = img

            # ต้นฉบับ
            write_image(Path(OUTPUT_DIR) / f"img{counter}.jpg", img_base)
            counter += 1

            # Augment
            for _ in range(AUG_PER_IMG):
                aug = pipeline(img_base)
                write_image(Path(OUTPUT_DIR) / f"img{counter}.jpg", aug)
                counter += 1

            if idx % 10 == 0 or idx == len(input_paths):
                print(f"[OK] {idx}/{len(input_paths)} : {src_path}")
        except Exception as e:
            print(f"[ERR] {src_path} -> {e}")

    print(f"✅ เสร็จแล้ว! สร้างไฟล์ {counter} ไฟล์ใน {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
