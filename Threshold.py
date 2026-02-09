import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Path โมเดล CNN ของคุณ ---
MODEL_PATH = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"
model = load_model(MODEL_PATH)

# --- Load color image ---
img = cv2.imread("digit_4.png")  # อ่านเป็น BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB

# --- Resize image ตาม input โมเดล (สมมติ 20x32) ---
digit_img = cv2.resize(img_rgb, (20,32))
digit_img = digit_img.astype('float32') / 255.0
digit_img = np.expand_dims(digit_img, axis=0)  # shape: (1,32,20,3)

# --- Compute pixel sum along Y-axis (ใช้ grayscale) ---
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
pixel_sum = np.sum(gray, axis=1)
max_val = np.max(pixel_sum)
phi = pixel_sum / max_val
phi_threshold = 0.5
phi_mean = np.mean(phi)

# --- Predict digit ---
pred = model.predict(digit_img)
pred_class = np.argmax(pred)  # 0-9 digits
pred_conf = np.max(pred)

# สมมติ class 10 = transition
transition_class = 10
if pred_class == transition_class:
    model_output = 'Transition'
else:
    model_output = pred_class


# --- Handle prev_digit + phi ---
prev_digit = 5  # สมมติเลขเริ่มต้น

if isinstance(model_output, int):
    prev_digit = model_output
    display_digit = model_output
else:  # transition
    if prev_digit is not None:
        # ใช้ phi mean ตัดสินเลขเปลี่ยน
        if phi_mean >= phi_threshold:
            display_digit = (prev_digit + 1) % 10
            prev_digit = display_digit
        else:
            display_digit = prev_digit
    else:
        # หากไม่มี prev_digit → กำหนดเลขเริ่มต้นชั่วคราว
        prev_digit = 0
        display_digit = prev_digit


if isinstance(model_output, int):
    prev_digit = model_output
    display_digit = model_output
else:
    display_digit = prev_digit if prev_digit is not None else np.nan
    if phi_mean >= phi_threshold and prev_digit is not None:
        display_digit = (prev_digit + 1) % 10
        prev_digit = display_digit

# --- แสดงผลใน terminal ---
print(f"Model output = {model_output}")
print(f"Phi mean = {phi_mean:.3f}")
print(f"Display digit = {display_digit}")

# --- Plot image และ pixel sum ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Digit Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.plot(pixel_sum, range(len(pixel_sum)), color="orange", label="Pixel Sum")
plt.axhline(phi_threshold*max_val, color="red", linestyle="--", label=f"Phi Threshold")
plt.gca().invert_yaxis()
plt.xlabel("Pixel Sum")
plt.ylabel("Y Position")
plt.legend()
plt.title(f"Pixel Sum vs Y Position\nDisplay digit: {display_digit}")

plt.tight_layout()
plt.show()
