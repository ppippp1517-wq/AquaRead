import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import Huber

# === CONFIG ===
IMAGE_FOLDER = "D:/projectCPE/resized_96x96"
CSV_FILE = "D:/projectCPE/dataset/labels/keypoints_annotated_normalized.csv"
IMAGE_SIZE = 96
EPOCHS = 50
BATCH_SIZE = 32
MODEL_OUTPUT = "D:/projectCPE/modelneedle.keras"

# === LOAD DATA ===
df = pd.read_csv(CSV_FILE)
X, y = [], []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype('float32') / 255.0
    X.append(img)
    y.append([row['cx'], row['cy'], row['px'], row['py']])

X = np.array(X)
y = np.array(y)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# === MODEL ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

# === CALLBACKS ===
checkpoint = ModelCheckpoint(MODEL_OUTPUT, save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# === TRAIN ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop]
)

# === PLOT LOSS ===
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\n\u2705 โมเดลบันทึกไว้ที่: {MODEL_OUTPUT}")
