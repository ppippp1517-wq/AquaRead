import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === CONFIG ===
MODEL_PATH = "D:/projectCPE/model_keypoint4.keras"
IMAGE_FOLDER = "D:/projectCPE/augmented_resized"
IMAGE_SIZE = 224

model = load_model(MODEL_PATH)

sample_images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))])[:5]

for fname in sample_images:
    path = os.path.join(IMAGE_FOLDER, fname)
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = img_resized.astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img_norm, axis=0))[0]

    # กลับเป็นพิกัดเดิม
    h, w = img.shape[:2]
    points = (pred * [w, h, w, h, w, h, w, h]).astype(int).reshape(4, 2)

    # วาดจุด
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"P{i+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {fname}")
    plt.axis('off')
    plt.show()
