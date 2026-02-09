import tensorflow as tf
from PIL import Image
import numpy as np
import glob, os, cv2

# ========== CONFIG ==========
MODEL_PATH = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"
TEST_IMAGE_DIR = r'D:\projectCPE\dataset\images\test'

# ปรับแต่งคอนทราสต์เล็กน้อย (ปรับค่าได้)
DO_CLAHE   = True
DO_UNSHARP = True
GAMMA      = 0.95   # <1 เข้มขึ้น, >1 สว่างขึ้น

# ========== LOAD MODEL ==========
model = tf.keras.models.load_model(MODEL_PATH)

def enhance_digit(pil_img: Image.Image) -> Image.Image:
    """ปรับคอนทราสต์/ความคม แล้วคืนเป็น RGB เดิม (ขนาดภาพไม่เปลี่ยน)"""
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

    rgb = np.dstack([gray, gray, gray])  # กลับเป็น 3 แชนเนล
    return Image.fromarray(rgb, mode='RGB')

# ========== LIST TEST FILES ==========
image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, '*.png')) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpeg'))

print(f"พบรูปทดสอบ {len(image_files)} ภาพ")

for img_path in image_files:
    # ========== LOAD & PREPROCESS (ปรับสี แต่แสดงผลแบบเดิม) ==========
    img = Image.open(img_path)
    img = enhance_digit(img)                # << เพิ่มบรรทัดนี้
    img_arr = np.array(img, dtype="float32")
    # ถ้าตอนเทรน normalize หาร 255 ให้เปิดบรรทัดนี้
    # img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # (1, H, W, 3)

    # ========== PREDICT ==========
    pred = model.predict(img_arr, verbose=0)
    class_idx = int(np.argmax(pred, axis=1)[0])
    class_name = str(class_idx) if class_idx < 10 else "NaN"

    print(f"{os.path.basename(img_path)} -> Predict: {class_name}   (probs: {np.round(pred[0],3)})")