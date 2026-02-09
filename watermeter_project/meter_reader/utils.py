# -*- coding: utf-8 -*-
import os
import re
import math
import glob
import shutil
import warnings

import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from django.conf import settings
from ultralytics import YOLO

# ===== DB models =====
from .models import Meter, MeterReading

warnings.filterwarnings("ignore", category=FutureWarning)

# ================= Output folders =================
CAPTURE_DIR = r"D:/projectCPE/dataset/images/capture_images"
DETECT_DIR = r"D:/projectCPE/dataset/images/detect_images"
CROP_DIR = r"D:/projectCPE/dataset/images/cropped_images"
CROP_CLASS0_DIR = os.path.join(CROP_DIR, "class0")        # (ไม่ใช้ตัด 1 box แล้ว แต่ยังเซฟ strip ได้)
CROP_CLASSX_DIR = os.path.join(CROP_DIR, "class1-3")
DIGIT_CROPS_DIR = os.path.join(CROP_CLASS0_DIR, "digits")  # 20x32 ต่อหลัก

SUMMARY_DIR = r"D:/projectCPE/dataset/images/summary_digit"    # ภาพสรุปฝั่งขวา
DETECT_DIGIT_DIR = r"D:/projectCPE/dataset/images/detect_digit"  # ภาพ detect ของ YOLO คลาสเดียว

for d in [
    CAPTURE_DIR, DETECT_DIR, CROP_DIR, CROP_CLASS0_DIR,
    CROP_CLASSX_DIR, DIGIT_CROPS_DIR, SUMMARY_DIR, DETECT_DIGIT_DIR
]:
    os.makedirs(d, exist_ok=True)

# =============== OPTIONAL DEPENDENCIES ===============
try:
    import cv2.ximgproc as xip  # noqa: F401
    _HAS_XIMGPROC = True
except Exception:
    _HAS_XIMGPROC = False

try:
    from skimage.morphology import skeletonize  # noqa: F401
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# =============== Models ===============
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# YOLO หลายคลาส (0 = digital strip, 1..3 = เข็ม) — ใช้แค่ 1..3 แล้ว
YOLO_MULTI_PATH = r"D:/backup/projectCPE/dataset/runs/detect/train12/weights/best.pt"
model_multi = YOLO(YOLO_MULTI_PATH)

# YOLO คลาสเดียว “digit”
YOLO_DIGIT_PATH = r"D:/backup/projectCPE/dataset_digital/runs/detect/digital_det2/weights/best.pt"
model_digit = YOLO(YOLO_DIGIT_PATH)

# CNN เข็ม (sin, cos)
CNN_MODEL_PATH = r"D:/backup/projectCPE/watermeter_project/meter_reader/modelneedle.h5"
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print(f" โหลดโมเดล CNN เข็ม '{CNN_MODEL_PATH}' สำเร็จ!")
except Exception as e:
    print(f" เกิดข้อผิดพลาดโหลดโมเดลเข็ม: {e}")
    cnn_model = None

# CNN ดิจิทัล (0..9 + NaN(=10))
DIGIT_MODEL_PATH = r"D:/backup/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"
PAIR_MODEL_PATH = r"D:/backup/projectCPE/pair_ab_keras.h5"
try:
    digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
    pair_model = tf.keras.models.load_model(PAIR_MODEL_PATH, compile=False)
    print(f" โหลดโมเดลดิจิทัล '{DIGIT_MODEL_PATH}' และ pair '{PAIR_MODEL_PATH}' สำเร็จ!")
except Exception as e:
    print(f" เกิดข้อผิดพลาดโหลดโมเดลดิจิทัล: {e}")
    digit_model, pair_model = None, None

# =============== Spec & preprocess ===============
def _get_digit_model_spec():
    if digit_model is None:
        return 32, 20, 3, "RGB"
    _, H, W, C = digit_model.input_shape
    color = "L" if C == 1 else "RGB"
    return int(H), int(W), int(C), color


def preprocess_for_digit_model(pil_img: Image.Image):
    H, W, C, color = _get_digit_model_spec()
    img = pil_img.convert(color).resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    elif C == 3 and arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    # NOTE: โมเดลดิจิทัลของคุณเทรนแบบ “ไม่หาร 255”
    NORMALIZE = False
    if NORMALIZE:
        arr = arr / 255.0
    return np.expand_dims(arr, 0)

# =============== Digit enhancement & rules ===============
DO_CLAHE = True
DO_UNSHARP = True
GAMMA = 0.95

DIGIT_CONF_MIN = 0.80
NAN_CONF_MIN = 0.50
PAIR_CONF_MIN = 0.50
MID_RATIO = 0.50

# เอนเอียงไปเลขน้อยกว่า: ลด threshold และเพิ่ม margin
AREA_THR = 0.30
PAIR_THR = {}            # ex. {6:0.68, 7:0.62}
IGNORE_X_MARGIN = 0.10
STRONG_PAIR_CONF = 0.95  # ต้องมั่นใจมากถึงจะยอมเลื่อนขึ้น
EXTRA_BELOW_MARGIN = 0.10

# ==== DB helpers (History) ====
def get_or_create_meter(meter_id: str) -> Meter:
    obj, _ = Meter.objects.get_or_create(meter_id=meter_id)
    return obj


def get_prev_reading_digits(meter: Meter):
    last = MeterReading.objects.filter(meter=meter).order_by("-timestamp").first()
    if not last:
        return None
    return {
        "x01": int(last.x01),
        "x001": int(last.x001),
        "x0001": int(last.x0001),
        "x00001": int(last.x00001),
    }

# Summary look
SRC_SCALE = 0.60
DIGIT_THUMB_H = 110
DIGIT_SPACING = 48
STRIP_LEFT_OFFSET = 120
TOTAL_OFFSET_UP = 18
FONT_DIGIT_PATH = r"C:/Windows/Fonts/arial.ttf"
FONT_TOTAL_PATH = r"C:/Windows/Fonts/arialbd.ttf"


def enhance_digit(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    if DO_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
    if DO_UNSHARP:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    if abs(GAMMA - 1.0) > 1e-6:
        inv = 1.0 / GAMMA
        table = (np.linspace(0, 1, 256) ** inv) * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        gray = table[gray]
    rgb = np.dstack([gray, gray, gray])
    return Image.fromarray(rgb).convert("RGB")


def to_dark_mask(g_uint8):
    _, mask = cv2.threshold(g_uint8, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(np.float32)


def dark_fraction_below(mask, mid_ratio=0.5, ignore_margin=0.10):
    H, W = mask.shape
    y_mid = int(round(mid_ratio * (H - 1)))
    x0 = int(round(ignore_margin * W))
    x1 = W - x0
    if x1 <= x0:
        x0, x1 = 0, W
    roi = mask[:, x0:x1]
    tot = roi.sum() + 1e-6
    below = roi[y_mid + 1 :, :].sum()
    return float(below / tot), y_mid


def text_size(draw, text, font):
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    return draw.textsize(text, font=font)

# =============== Analog (needle) ===============
def detect_needle_value(image_path):
    if cnn_model is None:
        return None
    img = Image.open(image_path).resize((32, 32)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    batch = np.reshape(arr, [1, 32, 32, 3])
    sin_val, cos_val = cnn_model.predict(batch, verbose=0)[0]
    rad = np.arctan2(sin_val, cos_val)
    return ((rad / (2 * math.pi)) % 1) * 10

# =============== Alignment ===============
class WaterMeterAligner:
    def __init__(self):
        self.detector = cv2.SIFT_create(nfeatures=8000)
        self.good_match_percent = 0.2

    def preprocess_meter_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)

    def detect_and_compute(self, image):
        return self.detector.detectAndCompute(self.preprocess_meter_image(image), None)

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(matcher.match(desc1, desc2), key=lambda x: x.distance)
        return matches[: int(len(matches) * self.good_match_percent)]

    def detect_circular_features(self, image):
        gray = self.preprocess_meter_image(image)
        return cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=50, maxRadius=300
        )

    def align_meter_images(self, reference_img, image_to_align):
        kp1, desc1 = self.detect_and_compute(reference_img)
        kp2, desc2 = self.detect_and_compute(image_to_align)
        matches = self.match_features(desc1, desc2)
        if len(matches) < 4:
            return image_to_align, None, len(matches), 0.0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
        if H is None:
            return image_to_align, None, len(matches), 0.0

        h, w = reference_img.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, H, (w, h))
        inliers = int(np.sum(mask)) if mask is not None else 0
        quality = inliers / max(1, len(matches))

        os.makedirs("alignment_results", exist_ok=True)
        cv2.imwrite("alignment_results/aligned_water_meter.jpg", aligned)
        np.save("alignment_results/transformation_matrix.npy", H)
        return aligned, H, len(matches), quality

# =============== Crop+Rotate (Hough circle) ===============
def crop_and_rotate_hough(image_path, rotation_angle=-5.0):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (9, 9), 2), 50, 150)

    min_r = int(min(h, w) * 0.3)
    max_r = int(min(h, w) * 0.7)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, 1.2, h // 4,
        param1=100, param2=60, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return img

    x, y, r = np.round(circles[0][0]).astype(int)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    top, bottom = max(y - r, 0), min(y + r, h)
    left, right = max(x - r, 0), min(x + r, w)

    circular_roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    circular_roi = np.where(mask[..., None] == 255, circular_roi, 255)
    roi = circular_roi[top:bottom, left:right]
    mask_cropped = mask[top:bottom, left:right]

    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(
        roi, M, (roi.shape[1], roi.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    mask_rot = cv2.warpAffine(
        mask_cropped, M, (mask_cropped.shape[1], mask_cropped.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return cv2.cvtColor(np.where(mask_rot[..., None] == 255, rotated, 255), cv2.COLOR_RGB2BGR)

# ====== Carry: ไล่จากขวา → ซ้าย เหมือนมิเตอร์จริง ======
def _wrapped_simple(prev_digit, curr_digit):
    """หาว่า wrap 9→0 หรือไม่ (ต้องมี prev ถึงจะชัวร์)"""
    if prev_digit is None:
        return False
    return int(prev_digit) == 9 and int(curr_digit) == 0


def _inc_digit(d):
    d = int(d) + 1
    carry = d == 10
    return (0 if carry else d), (1 if carry else 0)


def cascade_carry(integer, x01, x001, x0001, x00001, prev=None, ovf=None):
    I = int(integer) if isinstance(integer, str) else int(integer)
    d1 = int(x01)
    d2 = int(x001)
    d3 = int(x0001)
    d4 = int(x00001)

    pv = prev or {}
    ov = ovf or {}

    def _inc(d):
        d = int(d) + 1
        return (0 if d == 10 else d), (d == 10)

    def _wrap(prev_d, cur_d):
        return (prev_d is not None) and (int(prev_d) == 9 and int(cur_d) == 0)

    def _cascade_from_d4():
        nonlocal I, d1, d2, d3
        d3, c = _inc(d3)
        if c:
            d2, c = _inc(d2)
            if c:
                d1, c = _inc(d1)
                if c:
                    I += 1

    def _cascade_from_d3():
        nonlocal I, d1, d2
        d2, c = _inc(d2)
        if c:
            d1, c = _inc(d1)
            if c:
                I += 1

    def _cascade_from_d2():
        nonlocal I, d1
        d1, c = _inc(d1)
        if c:
            I += 1

    # ---- ใช้ OR: ovf | prev-wrap | pattern-wrap ----
    wrap_d4 = ov.get("x00001", False) or _wrap(pv.get("x00001"), d4) or (d4 == 0 and d3 == 9)
    if wrap_d4:
        _cascade_from_d4()

    wrap_d3 = ov.get("x0001", False) or _wrap(pv.get("x0001"), d3) or (d3 == 0 and d2 == 9 and d4 == 0)
    if wrap_d3:
        _cascade_from_d3()

    wrap_d2 = ov.get("x001", False) or _wrap(pv.get("x001"), d2) or (d2 == 0 and d1 == 9 and d3 == 0 and d4 == 0)
    if wrap_d2:
        _cascade_from_d2()

    wrap_d1 = ov.get("x01", False) or _wrap(pv.get("x01"), d1)
    if wrap_d1:
        I += 1

    return I, d1, d2, d3, d4

# =========================================================
# ดิจิทัล (แทน class0 เดิม): YOLO single-class + CNN + summary
# =========================================================
def run_digital_reader_on_aligned(aligned_bgr: np.ndarray, stem: str):
    """
    aligned_bgr: ภาพมิเตอร์ (BGR)
    stem: ชื่อไฟล์ฐาน (ไม่เอานามสกุล)
    return: (string_5digits, detect_path, summary_path)
    """
    if digit_model is None or pair_model is None:
        return "00000", None, None
    for old in glob.glob(os.path.join(DIGIT_CROPS_DIR, f"{stem}_*_20x32.png")):
        try:
            os.remove(old)
        except Exception:
            pass
    # 1) Detect digits (single-class YOLO)
    rs = model_digit.predict(
        source=aligned_bgr, imgsz=832, conf=0.05, iou=0.5,
        agnostic_nms=False, verbose=False
    )
    r = rs[0]
    if r.boxes is None or len(r.boxes) == 0:
        return "00000", None, None

    # save detect image
    det_bgr = r.plot()
    det_path = os.path.join(DETECT_DIGIT_DIR, f"{stem}_digit_detect.jpg")
    cv2.imwrite(det_path, det_bgr)

    # sort boxes left->right
        # --- get boxes + conf ---
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    # sort by x1 (left->right)
    order = np.argsort(boxes[:, 0])
    boxes = boxes[order]
    confs = confs[order]

    # ---- basic size filter (remove tiny/huge) ----
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    areas = ws * hs

    med_w = np.median(ws)
    med_h = np.median(hs)
    med_area = np.median(areas)

    keep = (
        (ws > 0.50 * med_w) & (ws < 1.80 * med_w) &
        (hs > 0.50 * med_h) & (hs < 1.80 * med_h) &
        (areas > 0.35 * med_area) & (areas < 2.50 * med_area)
    )
    boxes = boxes[keep]
    confs = confs[keep]

    # ---- if still > 5: pick best "row of digits" by y-center consistency ----
    if len(boxes) > 5:
        yc = (boxes[:, 1] + boxes[:, 3]) / 2.0
        y_med = np.median(yc)

        # digits should be roughly on same horizontal line
        # threshold based on median height
        y_thr = 0.45 * np.median(hs[keep]) if np.any(keep) else 15.0
        row_keep = np.abs(yc - y_med) <= y_thr

        # if row filter removes too much, fallback to original
        if row_keep.sum() >= 5:
            boxes = boxes[row_keep]
            confs = confs[row_keep]

    # ---- ensure at most 5: choose 5 highest conf, then re-sort left->right ----
    if len(boxes) > 5:
        top_idx = np.argsort(confs)[-5:]
        boxes = boxes[top_idx]
        confs = confs[top_idx]
        order = np.argsort(boxes[:, 0])
        boxes = boxes[order]
        confs = confs[order]

    # ---- if less than 5, just proceed (or you can pad later) ----
    boxes = boxes[:5]


    pil_src = Image.fromarray(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))
    Hm, Wm, Cm = digit_model.input_shape[1:4]
    Hp, Wp, Cp = pair_model.input_shape[1:4]

    preds = []
    crops_display = []
    paths_20x32 = []
    RESIZE_WH = (20, 32)

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        crop = pil_src.crop((x1, y1, x2, y2))
        base = f"{stem}_{i}"

        # เซฟ 20x32 (เก็บไว้ตรวจย้อนหลัง)
        crop_20x32 = crop.resize(RESIZE_WH, Image.BILINEAR)
        p_20x32 = os.path.join(DIGIT_CROPS_DIR, base + "_20x32.png")
        crop_20x32.save(p_20x32)
        crops_display.append(crop_20x32)
        paths_20x32.append(p_20x32)

        # 2) ใช้ pair model อย่างเดียวเพื่อเลือกเลขน้อย/เลขเดิม
        pil_enh = enhance_digit(crop_20x32)
        g_full = np.array(pil_enh.convert("L"))
        g_resz = cv2.resize(g_full, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        xpair = (g_resz.astype(np.float32) / 255.0)
        if Cp == 1:
            xpair = xpair[..., None]
        else:
            xpair = np.stack([xpair, xpair, xpair], axis=-1)

        pr = pair_model.predict(xpair[None, ...], verbose=0)[0]
        a = int(pr.argmax())  # เลขน้อย/เลขเดิม
        final = a
        preds.append(final)

    # 3) summary
    W0, H0 = pil_src.size
    W = int(W0 * SRC_SCALE)
    H = int(H0 * SRC_SCALE)
    pil_left = Image.fromarray(cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)

    thumbs = []
    for t in crops_display:
        th = DIGIT_THUMB_H
        tw = int(round(t.width * (th / t.height)))
        thumbs.append(t.resize((tw, th), Image.NEAREST))

    strip_w = sum(t.width for t in thumbs) + DIGIT_SPACING * (len(thumbs) - 1) if thumbs else 0
    strip_h = DIGIT_THUMB_H

    canvas_w = max(W + STRIP_LEFT_OFFSET + strip_w + 60, W + 600)
    canvas_h = max(H, strip_h + 120)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    canvas.paste(pil_left, (0, (canvas_h - H) // 2))
    d = ImageDraw.Draw(canvas)
    font_digit = ImageFont.truetype(FONT_DIGIT_PATH, 22)
    font_total = ImageFont.truetype(FONT_TOTAL_PATH, 40)

    total_text = "".join(str(x) for x in preds) if preds else "-----"
    tw, th = text_size(d, total_text, font_total)
    row_x = W + STRIP_LEFT_OFFSET
    row_y = (canvas_h - strip_h) // 2
    bx = row_x + (strip_w // 2) - (tw // 2)
    by = row_y - (th + 26) - TOTAL_OFFSET_UP
    d.rectangle([bx - 12, by - 12, bx + tw + 12, by + th + 12], outline=(20, 80, 200), width=3)
    d.text((bx, by), total_text, font=font_total, fill=(10, 10, 10))

    x = row_x
    y = row_y
    for i, t in enumerate(thumbs):
        canvas.paste(t, (x, y))
        lbl = str(preds[i]) if i < len(preds) else "?"
        ltw, lth = text_size(d, lbl, font_digit)
        d.rectangle([x, y - lth - 10, x + ltw + 10, y], fill=(0, 0, 0))
        d.text((x + 5, y - lth - 6), lbl, font=font_digit, fill=(255, 255, 255))
        x += t.width + DIGIT_SPACING

    summary_path = os.path.join(SUMMARY_DIR, f"{stem}_summary.jpg")
    canvas.save(summary_path, quality=92)

    return total_text, det_path, summary_path

# =========================================================
# Main pipeline
# =========================================================
def process_meter_image(image_path, meter_id="DEFAULT"):
    base_name = os.path.basename(image_path)
    captured_path = os.path.join(CAPTURE_DIR, base_name)
    try:
        if os.path.isfile(image_path):
            shutil.copy(image_path, captured_path)
    except Exception as e:
        print("[WARN] cannot save uploaded image:", e)

    print("\n=== WATER METER PIPELINE (NO CROP/ALIGN) START ===")

    # 0) โหลดภาพต้นฉบับ BGR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] Cannot read image:", image_path)
        return {
            "digital_x": "00000",
            "x001": "0",
            "x0001": "0",
            "x00001": "0",
            "total": "00000.000",
            "detected_image_path": None,
            "digit_detect_path": None,
            "digit_summary_path": None,
        }

    stem = os.path.splitext(base_name)[0]

    # 1) YOLO multi-class (เฉพาะ class 1..3)
    result_multi = model_multi.predict(source=img_bgr, conf=0.25, verbose=False)[0]
    yolo_path = os.path.join(DETECT_DIR, f"detected_{base_name}")
    cv2.imwrite(yolo_path, result_multi.plot())

    # 2) Digit strip (YOLO single-class + pair)
    int_part, digit_det_path, digit_sum_path = run_digital_reader_on_aligned(img_bgr, stem)
    if not int_part:
        int_part = "00000"

    # 3) เข็ม class1..3: เลือกกล่องที่ conf สูงสุดของแต่ละคลาส
    original_img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ocr_result_by_class = {}
    overflow_flags = {1: False, 2: False, 3: False}  # << อยู่ก่อน for

    if result_multi.boxes is None or len(result_multi.boxes) == 0:
        for class_id in [1, 2, 3]:
            ocr_result_by_class[class_id] = "0"
            overflow_flags[class_id] = False
    else:
        xyxy = result_multi.boxes.xyxy.cpu().numpy().astype(int)
        clsarr = result_multi.boxes.cls.cpu().numpy().astype(int)
        confs = result_multi.boxes.conf.cpu().numpy()

        for class_id in [1, 2, 3]:
            idxs = np.where(clsarr == class_id)[0]
            if idxs.size == 0:
                ocr_result_by_class[class_id] = "0"
                overflow_flags[class_id] = False
                continue

            best_local = idxs[np.argmax(confs[idxs])]
            x1, y1, x2, y2 = xyxy[best_local]

            crop = original_img_pil.crop((x1, y1, x2, y2)).convert("RGB")
            enhance = ImageEnhance.Contrast(crop).enhance(2.0)
            enhance.save(os.path.join(CROP_CLASSX_DIR, f"{stem}_class{class_id}.png"))

            temp = f"temp_crop_class{class_id}.png"
            enhance.save(temp)
            val = detect_needle_value(temp)

            if val is None:
                digit, ovf = 0, False
            else:
                q = int(np.round(float(val)))
                digit = q % 10          # 10 -> 0
                ovf = q == 10           # ธงว่าถูกปัดเป็น 10

            ocr_result_by_class[class_id] = str(digit)
            overflow_flags[class_id] = ovf

            try:
                os.remove(temp)
            except Exception:
                pass

    # ===== แยกจำนวนเต็ม + ทศนิยม (pos1..4) =====
    int_str = int_part or ""
    decimal0 = int_str[-1] if len(int_str) >= 1 else "0"  # x01 (pos1)
    int_only = int_str[:-1] if len(int_str) > 1 else "0"  # จำนวนเต็ม

    decimal1 = ocr_result_by_class.get(1, "0")  # x001
    decimal2 = ocr_result_by_class.get(2, "0")  # x0001
    decimal3 = ocr_result_by_class.get(3, "0")  # x00001

    # ==== prev จากฐานข้อมูล (ต้องมี ก่อนเรียก cascade_carry) ====
    meter_obj = get_or_create_meter(meter_id)
    prev_reading = get_prev_reading_digits(meter_obj)  # None ถ้ายังไม่มีประวัติ

    # --- เก็บ RAW ก่อน carry ---
    raw_digits = {
        "digital_x_raw": (int_only if int_only.lstrip("0") != "" else "0"),
        "x01_raw": str(decimal0),
        "x001_raw": str(decimal1),
        "x0001_raw": str(decimal2),
        "x00001_raw": str(decimal3),
    }

    # --- ส่งธง ovf จากเข็มเข้า cascade_carry ---
    ovf_map = {
        "x001": overflow_flags.get(1, False),
        "x0001": overflow_flags.get(2, False),
        "x00001": overflow_flags.get(3, False),
        # ถ้ามี logic ovf ให้ x01 ด้วย ค่อยเติม 'x01': True/False
    }

    I_after, d1_after, d2_after, d3_after, d4_after = cascade_carry(
        integer=(int_only if int_only.lstrip("0") != "" else "0"),
        x01=decimal0, x001=decimal1, x0001=decimal2, x00001=decimal3,
        prev=prev_reading,
        ovf=ovf_map,  # << เพิ่มอาร์กิวเมนต์นี้
    )

    # --- FINAL หลัง carry (+ เติม 0 นำหน้าให้เท่าความยาวของจำนวนเต็มที่ detect ได้) ---
    detected_len = len(int_part) if int_part else 0
    int_digits = max(detected_len - 1, 1)
    int_only_str = str(I_after).zfill(int_digits)

    decimal0 = str(d1_after)
    decimal1 = str(d2_after)
    decimal2 = str(d3_after)
    decimal3 = str(d4_after)

    final_digits = {
        "digital_x": int_only_str,
        "x01": decimal0,
        "x001": decimal1,
        "x0001": decimal2,
        "x00001": decimal3,
    }

    digit_string = int_only_str + decimal0
    combined = f"{int_only_str}.{decimal0}{decimal1}{decimal2}{decimal3}"

    # ===== ส่งไฟล์ไป MEDIA และคืนเป็น URL =====
    def _to_media(src_path: str):
        try:
            if not src_path or not os.path.isfile(src_path):
                return None
            out_dir = os.path.join(settings.MEDIA_ROOT, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, os.path.basename(src_path))
            if src_path != dst:
                shutil.copy(src_path, dst)
            return f"/media/outputs/{os.path.basename(src_path)}"
        except Exception as e:
            print("[WARN] copy to media failed:", e)
            return None

    detected_url = _to_media(yolo_path)
    digit_detect_url = _to_media(digit_det_path)
    digit_summary_url = _to_media(digit_sum_path)

    print("=== PIPELINE DONE (NO CROP/ALIGN) ===\n")

    # ==== บันทึกลง History ====
    try:
        MeterReading.objects.create(
            meter=meter_obj,
            digital_x=int_only_str,
            x01=int(decimal0), x001=int(decimal1), x0001=int(decimal2), x00001=int(decimal3),
            total=combined,
            detected_image_path=(detected_url or ""),
            digit_detect_path=(digit_detect_url or ""),
            digit_summary_path=(digit_summary_url or ""),
        )
    except Exception as e:
        print("[WARN] save reading failed:", e)

    return {
        "stem": stem,
        "digital_x": int_only_str,
        "x01": decimal0,
        "x001": decimal1,
        "x0001": decimal2,
        "x00001": decimal3,
        "total": combined,
        "digit_string": digit_string,
        "detected_image_path": detected_url,
        "digit_detect_path": digit_detect_url,
        "digit_summary_path": digit_summary_url,
        "prev_reading": prev_reading or {"x01": None, "x001": None, "x0001": None, "x00001": None},
        "raw_digits": raw_digits,       # ก่อน carry
        "final_digits": final_digits,   # หลัง carry
    }
