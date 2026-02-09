# D:/projectCPE/read_pipeline_from_roi.py
import os, sys, csv, argparse, importlib.util
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ============ DEFAULT PATHS (แก้ได้ผ่าน --args) ============
SEG_FILE     = r"D:/projectCPE/digit_segmentation_skeleton_cut (1).py"  # โค้ด split ที่คุณมี
MODEL_DIGIT  = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"  # 0..9 + NaN(=10)

ROI_IN       = r"D:/projectCPE/class1.png"                 # ROI จาก YOLO
ROI_OUT_DIR  = r"D:/projectCPE/out_digits"                 # เซฟ ROI หลัง post-process/deskew
CROP_DIR     = r"D:/projectCPE/dataset/images/cropnumber"  # เซฟ/ทับ 5 หลัก (ก่อน/หลัง resize)
SAVE_CSV     = r"D:/projectCPE/final_readout.csv"          # สรุปผล

# ============ โมเดลหลัก normalize ไหม ============
NORMALIZE_DIGIT = False   # ถ้าตอนเทรนหาร 255 ให้ตั้ง True

# --- Confidence thresholds (ตามโค้ดตัวอย่างของคุณ) ---
DIGIT_CONF_MIN = 0.80   # ถ้าโมเดลหลักมั่นใจ ≥ ค่านี้ ใช้ผลโมเดลหลักเลย
NAN_CONF_MIN   = 0.50   # จะถือว่าเป็นเคส transition/pending เมื่อ NaN_conf ≥ ค่านี้
# (ยังไม่ใช้ PAIR ในเวอร์ชันนี้ตามคำสั่ง “ตัดขั้นตอน 8 ออกก่อน”)

# ============ PHASE-related (ยังไม่ใช้ในเวอร์ชันนี้) ============
# คงค่าไว้เผื่อเปิด pair ในอนาคต
MID_RATIO = 0.50
AREA_THR  = 0.60
PAIR_THR  = {}
IGNORE_X_MARGIN = 0.10

# ---------------------------------------------------
def dyn_import_module(py_path, module_name="seg_mod"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None:
        raise RuntimeError(f"Cannot load module spec from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_img(path, img_bgr_or_gray):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img_bgr_or_gray)

def preprocess_to_model(pil_img: Image.Image, inshape, normalize=True):
    """
    ปรับขนาด/ช่องสีให้ตรง input ของโมเดล Keras (channels_last)
    return np.ndarray shape (1,H,W,C)
    """
    _, Hexp, Wexp, Cexp = inshape
    im = pil_img.resize((Wexp, Hexp), Image.BILINEAR)  # PIL: (W,H)
    arr = np.array(im, dtype=np.float32)
    if Cexp == 1 and arr.ndim == 3:
        # RGB -> GRAY
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and Cexp == 1:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    if normalize:
        arr = arr / 255.0
    return arr[None, ...]

# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser("YOLO ROI → post/deskew → split(5) → resize → read (main only; no pair)")
    ap.add_argument("--seg",   default=SEG_FILE)
    ap.add_argument("--roi",   default=ROI_IN)
    ap.add_argument("--roi_outdir", default=ROI_OUT_DIR)
    ap.add_argument("--cropdir",    default=CROP_DIR)
    ap.add_argument("--digit", default=MODEL_DIGIT)
    ap.add_argument("--csv",   default=SAVE_CSV)
    args = ap.parse_args()

    # ---------- 1) โหลดโมดูล split ----------
    seg = dyn_import_module(args.seg, "digit_seg_module")

    # ---------- 2) อ่าน ROI จาก YOLO ----------
    roi_bgr = cv2.imread(args.roi, cv2.IMREAD_COLOR)
    if roi_bgr is None:
        raise SystemExit(f"Cannot read ROI image: {args.roi}")

    # ---------- 3) post-process/deskew + แยก 5 หลัก ----------
    # segment_digits() ควรคืน [(gray_uint8, bin_uint8), ...] + meta ที่มีภาพหลัง deskew
    crops, meta = seg.segment_digits(roi_bgr, num_digits=5)

    # เซฟ ROI หลัง deskew (ตามที่ต้องการ)
    ensure_dir(args.roi_outdir)
    roi_after = meta.get("gray", cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY))
    save_img(os.path.join(args.roi_outdir, "roi_post_deskew.png"), roi_after)

    # ---------- 4) เซฟ 5 หลักที่ split ไปยัง CROP_DIR ----------
    ensure_dir(args.cropdir)
    for i, (g, b) in enumerate(crops):
        save_img(os.path.join(args.cropdir, f"digit_{i}.png"), g)

    # ---------- 5) โหลดโมเดลหลัก ----------
    model_digit = tf.keras.models.load_model(args.digit)   # 0..9 + NaN(10)
    inshape_digit = model_digit.input_shape                # (None,H,W,C)
    _, Hd, Wd, Cd = inshape_digit

    # ---------- 6) resize ให้ตรงกับโมเดล (ทับไฟล์เดิมใน CROP_DIR) ----------
    digits_paths = [os.path.join(args.cropdir, f"digit_{i}.png") for i in range(5)]
    for p in digits_paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise SystemExit(f"Missing crop: {p}")
        pil = Image.fromarray(g).convert("L" if Cd == 1 else "RGB")
        pil_sz = pil.resize((Wd, Hd), Image.BILINEAR)
        # เซฟทับเป็น GRAY หรือ BGR ตามที่อ่าน/เขียนสะดวก (ที่นี่เซฟ GRAY)
        save_img(p, np.array(pil_sz))

    # ---------- 7) อ่านเลขทีละหลัก (ใช้เกณฑ์ DIGIT_CONF_MIN + NAN_CONF_MIN; ยังไม่เรียก pair) ----------
    rows = [["idx","final","digit_pred","is_NaN","nan_conf","digit_conf","note"]]
    final_digits = []
    pending_exists = False

    for i, p in enumerate(digits_paths):
        pil = Image.open(p).convert("L" if Cd == 1 else "RGB")
        x = preprocess_to_model(pil, inshape_digit, normalize=NORMALIZE_DIGIT)
        probs = model_digit.predict(x, verbose=0)[0]

        cls        = int(np.argmax(probs))
        digit_conf = float(probs[cls])
        nan_idx    = 10 if probs.shape[0] == 11 else None
        nan_conf   = float(probs[nan_idx]) if nan_idx is not None else 0.0
        is_nan     = (nan_idx is not None and cls == nan_idx)

        use_main = (not is_nan) and (nan_conf < NAN_CONF_MIN or digit_conf >= DIGIT_CONF_MIN)

        if use_main:
            final_digits.append(cls)
            rows.append([i, cls, cls, 0, f"{nan_conf:.4f}", f"{digit_conf:.4f}", "main_ok"])
        else:
            # ยังไม่เรียก pair ตามคำสั่ง → ทำเครื่องหมาย pending/transition
            final_digits.append(-1)   # ใช้ -1 เป็นตัวแทน transition/ยังไม่ตัดสิน
            rows.append([i, "pending", ("NaN" if is_nan else cls), int(is_nan),
                         f"{nan_conf:.4f}", f"{digit_conf:.4f}", "need_pair_later"])
            pending_exists = True

    # ---------- 8) (ตัดออกตามคำสั่ง) pair + midline ----------
    # ไม่ทำในเวอร์ชันนี้

    # ---------- 9) แสดง/บันทึกผล ----------
    print("\n=== RESULT (5 digits) ===")
    print("digits (use -1 for pending):", final_digits)
    if (len(final_digits) >= 5) and (not pending_exists):
        int_part = final_digits[0:4]
        frac1    = final_digits[4]
        print(f"value: {''.join(map(str,int_part))}.{frac1}")
    else:
        print("value: (pending) บางหลักยังไม่ตัดสิน (ต้องเปิดขั้นตอน pair ภายหลัง)")

    if SAVE_CSV:
        ensure_dir(os.path.dirname(SAVE_CSV) or ".")
        with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("Wrote CSV ->", SAVE_CSV)

if __name__ == "__main__":
    main()
