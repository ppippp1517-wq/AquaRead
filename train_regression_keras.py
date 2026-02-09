import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

# === CONFIG ===
image_dir = "D:/projectCPE/dataset/images/cropped_resized"
csv_path = "D:/projectCPE/dataset/images/value_trues.csv"
img_size = (128, 128)

# === โหลด CSV ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=['value_true'])  # ลบภาพที่ยังไม่ได้ใส่ label

# === โหลดรูปภาพ ===
images = []
labels = []

for _, row in df.iterrows():
    path = os.path.join(image_dir, row['filename'])
    if os.path.exists(path):
        img = Image.open(path).resize(img_size).convert("RGB")
        img_arr = np.array(img) / 255.0
        images.append(img_arr)
        labels.append(float(row['value_true']))

X = np.array(images)
y = np.array(labels)

# === แบ่ง train/test ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === สร้างโมเดล CNN สำหรับ Regression ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output เดียวสำหรับค่า regression
])


model.compile(optimizer=Adam(1e-4), loss=MeanSquaredError(), metrics=['mae'])


# === เทรนโมเดล ===
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=16)

# === บันทึกโมเดล ===
model.save("regression_dial_model.keras")
print("บันทึกโมเดลเป็น regression_dial_model.keras")

# === แสดงกราฟผลลัพธ์ ===
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error')
plt.legend()
plt.grid()
plt.show()
