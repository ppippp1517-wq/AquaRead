import os, glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ========= CONFIG =========
MODEL_PATH = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"
TEST_IMAGE_DIR = r"D:\projectCPE\dataset\images\test"   # <== ถ้าเป็นรูป "ช่องตัวเลข" ให้เปลี่ยนเป็นโฟลเดอร์ cropnumber

# ========= LOAD MODEL =========
model = tf.keras.models.load_model(MODEL_PATH)
print("model.input_shape:", model.input_shape)  # (None, H, W, C)

# ดึง H,W,C จากโมเดล (channels_last)
_, Hexp, Wexp, Cexp = model.input_shape
COLOR_MODE = "L" if Cexp == 1 else "RGB"

def preprocess_pil(pil_img, show_preview=False, save_path=None):
    """แปลงสี/รีไซซ์/normalize ให้ตรงกับโมเดล และ (ถ้าขอ) แสดง/บันทึกภาพที่รีไซซ์แล้ว"""
    # 1) แปลงสีให้ตรงกับโมเดล
    pil_img = pil_img.convert(COLOR_MODE)

    # 2) รีไซซ์ให้ตรง (Wexp,Hexp) – PIL.resize รับ (width, height)
    resized = pil_img.resize((Wexp, Hexp), Image.BILINEAR)

    # 3) แสดง/บันทึกภาพรีไซซ์เพื่อเช็คด้วยตา
    if show_preview:
        plt.figure()
        plt.title(f"Resized to {Wexp}x{Hexp} ({COLOR_MODE})")
        plt.imshow(resized if COLOR_MODE=="RGB" else np.array(resized), cmap=None if COLOR_MODE=="RGB" else "gray")
        plt.axis('off')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            resized.save(save_path)
        plt.show(block=False)

    # 4) แปลงเป็น array + normalize
    arr = np.array(resized, dtype=np.float32)
    if Cexp == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, -1)   # (H,W,1)
    elif Cexp == 3 and arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    arr = arr / 255.0  # เปิดไว้ถ้าตอนเทรนหาร 255 (ส่วนใหญ่เป็นแบบนี้)
    arr = np.expand_dims(arr, 0)  # (1,H,W,C)
    return arr, resized

# ========= RUN =========
image_files = []
for ext in (".jpg",".jpeg","*.png"):
    image_files += glob.glob(os.path.join(TEST_IMAGE_DIR, ext))

print(f"พบรูปทดสอบ {len(image_files)} ภาพ")
for img_path in sorted(image_files):
    pil = Image.open(img_path)
    x, resized = preprocess_pil(
        pil, 
        show_preview=True, 
        save_path=os.path.join(os.path.dirname(img_path), "_resized_preview", os.path.basename(img_path))
    )

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    # ถ้าใช้คลาส transition=10 ให้ map ชื่อคลาสตามนี้
    class_names = [str(i) for i in range(10)] + ["transition"] if probs.shape[0] == 11 else [str(i) for i in range(probs.shape[0])]
    pred_name = class_names[idx]

    # top-3 ให้ดู confidence
    topk = np.argsort(probs)[::-1][:3]
    top_str = ", ".join([f"{class_names[k]}={probs[k]:.2f}" for k in topk])

    print(f"{os.path.basename(img_path)} -> Predict: {pred_name} | top3: {top_str}")