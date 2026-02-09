import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# === CONFIG ===
image_dir = "D:/projectCPE/dataset/images/crop"
csv_path = "D:/projectCPE/dataset/images/angle_labels.csv"
img_size = (128, 128)

# === Load CSV ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=['angle'])  # ลบแถวที่ไม่มี label

# === Load Images ===
images = []
angles = []

for _, row in df.iterrows():
    fname = row['filename']
    angle = row['angle']

    found = False
    for ext in ['.png', '.jpg', '.jpeg']:
        path = os.path.join(image_dir, fname + ext)
        if os.path.exists(path):
            img = Image.open(path).resize(img_size).convert("RGB")
            img_arr = np.array(img) / 255.0
            images.append(img_arr)
            angles.append(float(angle))
            found = True
            break
    if not found:
        print(f"❌ ไม่พบไฟล์ภาพสำหรับ: {fname}")


X = np.array(images)
y = np.array(angles)

# === Train/Test Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === สร้างโมเดล CNN สำหรับ Regression ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output เป็นค่า angle เดียว
])

model.compile(optimizer=Adam(1e-4), loss=MeanSquaredError(), metrics=['mae'])

# === เทรน ===
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=16)

# === บันทึกโมเดล ===
model.save("regression_angle_model.keras")
print("✅ บันทึกโมเดลเป็น regression_angle_model.keras")

# === แสดงกราฟ MAE ===
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("Mean Absolute Error (Angle Prediction)")
plt.legend()
plt.grid()
plt.show()
