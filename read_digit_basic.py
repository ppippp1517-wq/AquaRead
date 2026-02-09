import os, glob, re
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ========== CONFIG ==========
MODEL_PATH     = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"  # โมเดลหลัก 0..9 + NaN(=10)
MODEL_PAIR     = r"D:/projectCPE/pair_ab_keras.h5"                             # โมเดลคู่เลข a->b
TEST_IMAGE_DIR = r"D:/projectCPE/dataset/images/test"

# ปรับภาพ
DO_CLAHE   = True
DO_UNSHARP = True
GAMMA      = 0.95
NORMALIZE  = False   # โมเดลหลักของคุณเทรนแบบไม่หาร 255 → False

# เกณฑ์ตัดสิน
DIGIT_CONF_MIN = 0.80   # ถ้าโมเดลหลักมั่นใจ ≥ ค่านี้ ใช้ผลโมเดลหลักได้เลย
NAN_CONF_MIN   = 0.50   # จะเรียก pair เมื่อ NaN_conf ≥ ค่านี้ (หรือทายเป็น NaN ตรง ๆ)
PAIR_CONF_MIN  = 0.50   # conf ของโมเดล pair ต้อง ≥ ค่านี้ด้วย

MID_RATIO      = 0.50   # เส้นแบ่งกลาง (0..1)
AREA_THR       = 0.60   # สัดส่วนมวลความมืดใต้เส้น ≥ นี้ ⇒ เอา b
PAIR_THR       = {      # เกณฑ์เฉพาะคู่ (ใส่เฉพาะคู่ที่อยากเข้มขึ้น)
    # 6: 0.68, 7: 0.62,
}
IGNORE_X_MARGIN = 0.10  # ตัดขอบซ้าย/ขวาทิ้งตอนนับพื้นที่

SAVE_PLOTS = True
PLOT_DIR   = r"D:/projectCPE/plots_transition"
PLOT_DPI   = 150

# ========== UTILS ==========
def enhance_digit(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert('L'))
    if DO_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
    if DO_UNSHARP:
        blur = cv2.GaussianBlur(gray, (0,0), 1.0)
        gray = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    if abs(GAMMA - 1.0) > 1e-6:
        inv = 1.0 / GAMMA
        table = (np.linspace(0,1,256) ** inv) * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        gray  = table[gray]
    rgb = np.dstack([gray, gray, gray])
    # เลี่ยง Deprecation: ใช้ convert("RGB") แทน mode='RGB'
    return Image.fromarray(rgb).convert("RGB")

def natural_sort_key(path):
    name = os.path.basename(path)
    nums = re.findall(r'\d+', name)
    return [int(n) for n in nums] if nums else [name.lower()]

def to_dark_mask(g_uint8):
    _, mask = cv2.threshold(g_uint8, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(np.float32)

def dark_fraction_below(mask, mid_ratio=0.5, ignore_margin=0.10):
    H, W = mask.shape
    y_mid = int(round(mid_ratio*(H-1)))
    x0 = int(round(ignore_margin*W)); x1 = W - x0
    if x1 <= x0: x0, x1 = 0, W
    roi = mask[:, x0:x1]
    tot = roi.sum() + 1e-6
    below = roi[y_mid+1:, :].sum()
    return float(below/tot), y_mid

def plot_pair_midline(gray_resz, a,b,conf, frac, thr, ymid, out_path):
    H,W = gray_resz.shape
    mask = to_dark_mask(gray_resz)
    row_dark = mask.sum(axis=1); y = np.arange(H)
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(cv2.resize(gray_resz, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2RGB))
    ax1.axis("off"); ax1.set_title(f"{a}->{b} conf={conf:.2f}")
    ax1.axhline(ymid*4, color='C0', lw=3)  # คูณ 4 เพราะรูปซ้ายถูกขยาย
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(row_dark, y, label="Row darkness"); ax2.invert_yaxis()
    ax2.axhline(ymid, color='C0', lw=2)
    ax2.set_xlabel("Darkness"); ax2.set_ylabel("Y")
    ax2.set_title(f"area_below={frac:.3f} | thr={thr:.2f}")
    ax2.legend(); fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close(fig)

# ========== LOAD MODELS ==========
model = tf.keras.models.load_model(MODEL_PATH)      # main
pair  = tf.keras.models.load_model(MODEL_PAIR, compile=False)  # pair

# shapes
Hm, Wm, Cm = model.input_shape[1:4]
Hp, Wp, Cp = pair.input_shape[1:4]

# ========== LIST FILES ==========
exts = ("*.jpg","*.png","*.jpeg","*.bmp","*.tif","*.tiff")
image_files = []
for e in exts: image_files += glob.glob(os.path.join(TEST_IMAGE_DIR, e))
image_files = sorted(image_files, key=natural_sort_key)
print(f"พบรูปทดสอบ {len(image_files)} ภาพ")

for img_path in image_files:
    # ---------- main model ----------
    pil0 = Image.open(img_path)
    pil  = enhance_digit(pil0)                 # RGB (enhanced)
    pil_m = pil.resize((Wm, Hm), Image.BILINEAR)
    arr = np.array(pil_m, dtype="float32")
    if Cm == 1:  # ถ้าโมเดลหลักรับ 1 แชนเนล
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)[...,None]
    if NORMALIZE: arr = arr/255.0
    x = arr[None,...]
    probs = model.predict(x, verbose=0)[0]

    cls        = int(np.argmax(probs))
    digit_conf = float(probs[cls])
    # เช็คว่ามีคลาส NaN ไหม (สมมติ index=10)
    nan_idx  = 10 if probs.shape[0] == 11 else None
    nan_conf = float(probs[nan_idx]) if nan_idx is not None else 0.0
    is_nan   = (nan_idx is not None and cls == nan_idx)

    # ---------- gating: ใช้ผลโมเดลหลักถ้าไม่ใช่ NaN และ NaN_conf ต่ำ ----------
    use_main = (not is_nan) and (nan_conf < NAN_CONF_MIN or digit_conf >= DIGIT_CONF_MIN)
    if use_main:
        print(f"{os.path.basename(img_path)} -> Predict: {cls}   (probs: {np.round(probs,3)})")
        continue

    # ---------- pair + midline ----------
    g_full = np.array(pil.convert("L"))  # ใช้ภาพ enhanced เดิม
    g_resz = cv2.resize(g_full, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    xpair = (g_resz.astype(np.float32)/255.0)
    if Cp == 1: xpair = xpair[...,None]
    else:       xpair = np.stack([xpair,xpair,xpair], axis=-1)
    p = pair.predict(xpair[None,...], verbose=0)[0]

    a = int(p.argmax()); b = (a+1)%10; conf=float(p[a])
    mask = to_dark_mask(g_resz)
    frac, ymid = dark_fraction_below(mask, MID_RATIO, IGNORE_X_MARGIN)
    thr_used = PAIR_THR.get(a, AREA_THR)
    final = b if (conf >= PAIR_CONF_MIN and frac >= thr_used) else a

    print(f"{os.path.basename(img_path)} -> Predict: NaN/transition (main={cls}, "
          f"nan_conf={nan_conf:.2f}, main_conf={digit_conf:.2f}) | "
          f"pair {a}->{b} conf={conf:.2f} | area_below={frac:.3f} thr={thr_used:.2f} => final={final}")

    if SAVE_PLOTS:
        out_plot = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(img_path))[0] + "_midline.png")
        plot_pair_midline(g_resz, a,b,conf, frac, thr_used, ymid, out_plot)
