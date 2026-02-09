#ครอบเลขแล้วทดสอบการเทรนเลขดิจิทัล
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ========= PATHS =========
FULL_IMAGE_PATH = r"D:\projectCPE\dataset\images\test\meter_full.jpg"  # <<< ใส่ภาพมิเตอร์เต็ม
YOLO_MODEL_PATH = r"D:/projectCPE/dataset/runs/detect/train12/weights/best.pt"

SAVE_PANEL_DIR  = r"D:\projectCPE\dataset\images\cropped_images"
SAVE_SLOTS_DIR  = r"D:\projectCPE\dataset\images\cropnumber"

# ========= CONFIG & LOAD DIGITAL MODEL =========
DIGIT_MODEL_PATH   = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"
DIGIT_IMG_SIZE     = (32, 20)   # ให้ตรงกับตอนเทรน (W,H สำหรับ PIL.resize)
DIGIT_COLOR_MODE   = "RGB"      # ถ้าเทรน gray ใช้ "L"
DIGIT_NORMALIZE_01 = True       # ถ้าตอนเทรนหาร 255
CLASS_NAMES        = [str(i) for i in range(10)] + ["transition"]
TRANSITION_INDEX   = 10
TRANSITION_THRESH  = 0.50
N_DIGITS           = 5

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

try:
    digit_classifier_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
    print(f" โหลดโมเดล Digital CNN '{DIGIT_MODEL_PATH}' สำเร็จ!")
except Exception as e:
    print(f" โหลดโมเดลดิจิทัลไม่สำเร็จ: {e}")
    raise

# ========= HELPER FUNCTIONS =========
def _preprocess_digit(img_pil: Image.Image):
    img_pil = img_pil.convert("L" if DIGIT_COLOR_MODE=="L" else "RGB")
    img_pil = img_pil.resize(DIGIT_IMG_SIZE, Image.BILINEAR)
    arr = np.array(img_pil, dtype=np.float32)
    if DIGIT_NORMALIZE_01:
        arr = arr / 255.0
    if arr.ndim == 2:  # gray
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)  # (1,H,W,C)

def _predict_digit_raw(img_pil: Image.Image):
    x = _preprocess_digit(img_pil)
    probs = digit_classifier_model.predict(x, verbose=0)[0]  # (11,)
    return int(np.argmax(probs)), probs

def _bottom_ink_ratio(slot_pil: Image.Image):
    gray = np.array(slot_pil.convert("L"))
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h = th.shape[0]
    bottom_half = th[h//2:, :]
    total_ink   = np.sum(th) / 255.0
    bottom_ink  = np.sum(bottom_half) / 255.0
    return 0.0 if total_ink <= 1e-6 else float(bottom_ink / total_ink)

def _resolve_transition(slot_pil: Image.Image, probs):
    # base = top-1 ใน 0..9, next = base+1 (mod 10). ถ้า bottom_ratio >= 0.5 → เลือก next
    order = np.argsort(probs[:10])[::-1]
    base_digit = int(order[0])
    next_digit = (base_digit + 1) % 10
    ratio = _bottom_ink_ratio(slot_pil)
    final_digit = next_digit if ratio >= TRANSITION_THRESH else base_digit
    return str(final_digit)

def predict_digit_with_rule(slot_pil: Image.Image):
    idx, probs = _predict_digit_raw(slot_pil)
    if idx != TRANSITION_INDEX:
        return str(idx)
    else:
        return _resolve_transition(slot_pil, probs)

def split_panel_into_slots(panel_pil: Image.Image, n=N_DIGITS):
    W, H = panel_pil.size
    dw = W // n
    return [panel_pil.crop((i*dw, 0, (i+1)*dw if i < n-1 else W, H)) for i in range(n)]

# ========= YOLO DETECTION & DIGIT READING =========
def main():
    # 1) โหลด YOLO + ภาพเต็ม
    model = YOLO(YOLO_MODEL_PATH)
    aligned_img = cv2.imread(FULL_IMAGE_PATH)
    if aligned_img is None:
        raise RuntimeError(f"อ่านภาพไม่ได้: {FULL_IMAGE_PATH}")

    # 2) รัน YOLO
    results = model.predict(source=FULL_IMAGE_PATH, conf=0.25)[0]
    df = results.to_df()

    # 3) เลือกกล่อง class 0 ที่มั่นใจที่สุด
    digital_rows = df[df['class'] == 0]
    if digital_rows.empty:
        print("ไม่พบกล่อง class 0")
        return

    best0 = digital_rows.loc[digital_rows['confidence'].idxmax()]
    x1,y1,x2,y2 = int(best0['box']['x1']), int(best0['box']['y1']), int(best0['box']['x2']), int(best0['box']['y2'])

    # 4) ครอปแผงดิจิทัล และเซฟ
    base = os.path.splitext(os.path.basename(FULL_IMAGE_PATH))[0]
    ensure_dir(SAVE_PANEL_DIR)
    ensure_dir(SAVE_SLOTS_DIR)

    panel_bgr = aligned_img[y1:y2, x1:x2]
    panel_path = os.path.join(SAVE_PANEL_DIR, f"{base}_panel.png")
    cv2.imwrite(panel_path, panel_bgr)
    print(f"✓ Saved panel: {panel_path}")

    # 5) แบ่ง 5 ช่อง และเซฟช่องละรูป
    panel_pil = Image.fromarray(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
    slots = split_panel_into_slots(panel_pil, n=5)

    slot_paths = []
    for i, slot in enumerate(slots):
        sp = os.path.join(SAVE_SLOTS_DIR, f"{base}_slot_{i}.png")
        slot.save(sp)
        slot_paths.append(sp)
    print("✓ Saved slots:", slot_paths)

    # 6) อ่านเลขด้วย rule 50%
    pred_digits = [predict_digit_with_rule(slot) for slot in slots]
    result_str = "".join(pred_digits)
    print("เลขดิจิทัล 5 หลัก:", result_str)

if __name__ == "__main__":
    main()
