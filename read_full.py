# -*- coding: utf-8 -*-
"""
Pipeline (per image):
1) YOLO (single-class 'digit') -> detect all boxes -> crop L->R
   - Save ROI full      -> CROP_FULL_DIR
   - Save resized 20x32 -> RESIZE_DIR
   - Save detected image (with boxes) -> DETECT_DIR
2) CNN reader (MODEL_PATH + MODEL_PAIR) on RESIZE_DIR images
3) Compose a summary image:
   - Left: detected image (scaled to 60%)
   - Right: a row of enlarged digit crops with predicted labels above (not overlapping)
   - Total number in a blue box ABOVE the row (centered), adjustable with TOTAL_OFFSET_UP
"""

import os, re, glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ------------------- PATHS & CONFIG -------------------
# YOLO
YOLO_MODEL_PATH  = r'D:/projectCPE/dataset_digital/runs/detect/digital_det2/weights/best.pt'
TEST_DIR         = r'D:/projectCPE/dataset/images/test'
CROP_FULL_DIR    = r'D:/projectCPE/dataset/images/cropdigital_images'
RESIZE_DIR       = r'D:/projectCPE/dataset/images/resize_crop'
DETECT_DIR       = r'D:/projectCPE/dataset/images/detect_images'      # ภาพที่มีกรอบดีเทค
SUMMARY_DIR      = r'D:/projectCPE/dataset/images/summary'            # ภาพสรุป

RESIZE_WH        = (20, 32)  # (width, height) สำหรับภาพครอปย่อ

# Summary canvas controls
SRC_SCALE          = 0.60   # ย่อภาพซ้าย (ดีเทค) ให้เหลือ 60%
DIGIT_THUMB_H      = 120    # ความสูงตัวเลขในแถวด้านขวา (ใหญ่ขึ้น)
DIGIT_SPACING      = 48     # ช่องว่างระหว่างภาพตัวเลข (มากขึ้น)
STRIP_LEFT_OFFSET  = 120    # ระยะจากขอบขวาของภาพซ้าย -> แถวตัวเลข
TOTAL_OFFSET_UP    = 25     # (+) ขยับกรอบสีน้ำเงิน “ขึ้น” จากตำแหน่งปกติ (พิกเซล)

# Fonts (Windows)
FONT_MAIN_PATH  = r'C:/Windows/Fonts/arial.ttf'
FONT_DIGIT_PATH = r'C:/Windows/Fonts/arial.ttf'
FONT_TOTAL_PATH = r'C:/Windows/Fonts/arialbd.ttf'

# Digit Reader (Keras/TensorFlow)
MODEL_PATH     = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"  # 0..9 + NaN(=10)
MODEL_PAIR     = r"D:/projectCPE/pair_ab_keras.h5"                              # a->b transition

# Image enhancement (for reader)
DO_CLAHE   = True
DO_UNSHARP = True
GAMMA      = 0.95
NORMALIZE  = False  # โมเดลหลักเทรนแบบไม่หาร 255

# Gating rules
DIGIT_CONF_MIN = 0.80
NAN_CONF_MIN   = 0.50
PAIR_CONF_MIN  = 0.50
MID_RATIO      = 0.50
AREA_THR       = 0.60
PAIR_THR       = {}        # ex. {6:0.68, 7:0.62}
IGNORE_X_MARGIN = 0.10
SAVE_PLOTS = False

# -------------- UTILS --------------
def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def is_image(p): 
    return p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))

def natural_key(path):
    """safe natural sort key: returns tuple of (tag, value) to avoid int<->str compare"""
    b = os.path.basename(path)
    parts = re.split(r'(\d+)', b)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p)))
        else:
            key.append((0, p.lower()))
    return tuple(key)

def text_size(draw, text, font):
    # Pillow ใหม่ใช้ getbbox แทน textsize
    if hasattr(font, "getbbox"):
        l,t,r,b = font.getbbox(text)
        return r-l, b-t
    return draw.textsize(text, font=font)

# ---------- Reader helpers ----------
import tensorflow as tf
import matplotlib.pyplot as plt

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
    return Image.fromarray(rgb).convert("RGB")

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

# -------------- YOLO stage --------------
def yolo_stage():
    ensure_dirs(CROP_FULL_DIR, RESIZE_DIR, DETECT_DIR, SUMMARY_DIR)
    yolo = YOLO(YOLO_MODEL_PATH)

    img_list = sorted(
        [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if is_image(f)],
        key=natural_key
    )
    outputs = []  # one dict per source image

    print("== YOLO detect & crop ==")
    for src in img_list:
        stem = os.path.splitext(os.path.basename(src))[0]
        rs = yolo.predict(source=src, imgsz=832, conf=0.05, iou=0.5,
                          agnostic_nms=False, verbose=False)
        r = rs[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(f"   ❌ {os.path.basename(src)}: no boxes")
            continue

        # detected image (left image)
        det_bgr = r.plot()
        det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
        pil_left = Image.fromarray(det_rgb)
        cv2.imwrite(os.path.join(DETECT_DIR, stem + "_detect.jpg"), det_bgr)

        # crop all boxes (sorted L->R)
        boxes = r.boxes.xyxy.cpu().numpy()
        order = np.argsort(boxes[:,0])
        boxes = boxes[order]

        pil_src = Image.open(src).convert("RGB")
        crops_display = []   # thumbnails (20x32 -> upscale later)
        paths_20x32 = []

        for i,(x1,y1,x2,y2) in enumerate(boxes, start=1):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            crop = pil_src.crop((x1,y1,x2,y2))
            base = f"{stem}_{i}"

            # ROI เต็ม
            p_full = os.path.join(CROP_FULL_DIR, base + "_roi.png")
            crop.save(p_full)

            # 20x32 for reader
            crop_20x32 = crop.resize(RESIZE_WH, Image.BILINEAR)
            p_20x32 = os.path.join(RESIZE_DIR, base + "_20x32.png")
            crop_20x32.save(p_20x32)
            paths_20x32.append(p_20x32)
            crops_display.append(crop_20x32)

        outputs.append({
            "stem": stem,
            "pil_left": pil_left,        # detected image
            "paths_20x32": paths_20x32,
            "crops_display": crops_display
        })
        print(f"   ✓ {os.path.basename(src)}: saved {len(boxes)} crops")
    return outputs

# -------------- Reader stage + Summary --------------
def reader_stage(yolo_outs):
    model = tf.keras.models.load_model(MODEL_PATH)
    pair  = tf.keras.models.load_model(MODEL_PAIR, compile=False)
    Hm, Wm, Cm = model.input_shape[1:4]
    Hp, Wp, Cp = pair.input_shape[1:4]

    # fonts
    font_digit = ImageFont.truetype(FONT_DIGIT_PATH, 22)
    font_total = ImageFont.truetype(FONT_TOTAL_PATH, 40)

    for out in yolo_outs:
        stem          = out["stem"]
        pil_left_full = out["pil_left"]
        paths_20x32   = out["paths_20x32"]
        crops_display = out["crops_display"]

        # --- predict each crop ---
        preds = []
        for p in paths_20x32:
            pil0 = Image.open(p)
            pil  = enhance_digit(pil0)
            pil_m = pil.resize((Wm, Hm), Image.BILINEAR)
            arr = np.array(pil_m, dtype="float32")

            if Cm == 1:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)[...,None]
            if NORMALIZE:
                arr = arr/255.0

            probs = model.predict(arr[None,...], verbose=0)[0]
            cls = int(np.argmax(probs))
            digit_conf = float(probs[cls])
            nan_idx  = 10 if probs.shape[0] == 11 else None
            nan_conf = float(probs[nan_idx]) if nan_idx is not None else 0.0
            is_nan   = (nan_idx is not None and cls == nan_idx)

            if (not is_nan) and (nan_conf < NAN_CONF_MIN or digit_conf >= DIGIT_CONF_MIN):
                final = cls
            else:
                g_full = np.array(pil.convert("L"))
                g_resz = cv2.resize(g_full, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
                xpair  = (g_resz.astype(np.float32)/255.0)
                if Cp == 1: xpair = xpair[...,None]
                else:       xpair = np.stack([xpair,xpair,xpair], axis=-1)
                pr = pair.predict(xpair[None,...], verbose=0)[0]
                a = int(pr.argmax()); b = (a+1)%10; conf=float(pr[a])
                mask = to_dark_mask(g_resz)
                frac, _ = dark_fraction_below(mask, MID_RATIO, IGNORE_X_MARGIN)
                thr_used = PAIR_THR.get(a, AREA_THR)
                final = b if (conf >= PAIR_CONF_MIN and frac >= thr_used) else a

            preds.append(final)

        # ====== Compose summary image ======
        # left image (detected) scaled
        W0,H0 = pil_left_full.size
        W = int(W0 * SRC_SCALE); H = int(H0 * SRC_SCALE)
        pil_left = pil_left_full.resize((W, H), Image.LANCZOS)

        # thumbnails row (bigger)
        thumb_imgs = []
        for t in crops_display:
            th = DIGIT_THUMB_H
            tw = int(round(t.width * (th / t.height)))
            thumb_imgs.append(t.resize((tw, th), Image.NEAREST))

        strip_w = sum(t.width for t in thumb_imgs) + DIGIT_SPACING*(len(thumb_imgs)-1) if thumb_imgs else 0
        strip_h = DIGIT_THUMB_H

        canvas_w = max(W + STRIP_LEFT_OFFSET + strip_w + 60, W + 600)
        canvas_h = max(H, strip_h + 120)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (245,245,245))

        # paste left (detected)
        canvas.paste(pil_left, (0, (canvas_h - H)//2))

        # draw helpers
        d = ImageDraw.Draw(canvas)

        # total text (blue box ABOVE the row, centered)
        total_text = ''.join(str(x) for x in preds) if preds else '----'
        tw, th = text_size(d, total_text, font_total)

        # anchor row position first
        row_x = W + STRIP_LEFT_OFFSET
        row_y = (canvas_h - strip_h)//2

        bx = row_x + (strip_w // 2) - (tw // 2)        # center above row
        by = row_y - (th + 26) - TOTAL_OFFSET_UP       # move UP by TOTAL_OFFSET_UP

        d.rectangle([bx-12, by-12, bx+tw+12, by+th+12], outline=(20,80,200), width=3)
        d.text((bx, by), total_text, font=font_total, fill=(10,10,10))

        # digits row (right side)
        x = row_x
        y = row_y
        for i, t in enumerate(thumb_imgs):
            canvas.paste(t, (x, y))
            # label above each digit (black box)
            lbl = str(preds[i]) if i < len(preds) else '?'
            ltw, lth = text_size(d, lbl, font_digit)
            d.rectangle([x, y- lth - 10, x + ltw + 10, y], fill=(0,0,0))
            d.text((x+5, y - lth - 6), lbl, font=font_digit, fill=(255,255,255))
            x += t.width + DIGIT_SPACING

        # save summary
        out_path = os.path.join(SUMMARY_DIR, f"{stem}_summary.jpg")
        canvas.save(out_path, quality=92)
        print(f" -> {stem}: total={total_text} | saved summary: {out_path}")

# ------------------- RUN -------------------
if __name__ == "__main__":
    outs = yolo_stage()
    if not outs:
        print("No detections. Done.")
    else:
        reader_stage(outs)
        print("\nAll done ✓")
