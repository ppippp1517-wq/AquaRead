import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# === CONFIG ===
model_path = "D:/projectCPE/regression_dial_model.keras"  # แก้ path ให้ตรงไฟล์ใหม่
image_dir = "D:/projectCPE/dataset/images/cropped_resized"  # โฟลเดอร์ภาพ dial ที่ resize แล้ว
img_size = (128, 128)

# === โหลดโมเดล ===
model = load_model(model_path)

# === ทำ prediction ===
for fname in os.listdir(image_dir):
    if fname.endswith(".png"):
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).resize(img_size).convert("RGB")
        img_arr = np.array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)  # เพิ่ม batch dimension

        prediction = model.predict(img_arr)[0][0]  # ค่าเดียว
        print(f"{fname}: {round(prediction, 2)}")
