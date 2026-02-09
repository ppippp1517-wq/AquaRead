import tensorflow as tf
import numpy as np
import math
from PIL import Image

# --- 1. ตั้งค่าไฟล์ ---
MODEL_PATH = 'CNN_Analog-Readout_Version-5.0.0.h5'
IMAGE_PATH = 'D:/projectCPE/dataset/images/cropped_images/class1_img1.jpg_2.png' # <== เปลี่ยนเป็นชื่อไฟล์รูปภาพของคุณ

# --- 2. โหลดโมเดลที่ฝึกไว้แล้ว ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✔️ โหลดโมเดล '{MODEL_PATH}' สำเร็จ!")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    exit()

# --- 3. โหลดและเตรียมรูปภาพ ---
# โมเดลนี้ถูกฝึกด้วยรูปภาพสี (RGB) ขนาด 32x32 pixels
try:
    # เปิดรูปภาพ
    img = Image.open(IMAGE_PATH)
    # ปรับขนาดรูปภาพให้เป็น 32x32
    img = img.resize((32, 32))
    # แปลงรูปภาพเป็น Numpy array
    image_array = np.array(img, dtype="float32")
    # เพิ่มมิติของ array สำหรับ Batch (โมเดลรับข้อมูลเป็นชุดเสมอ)
    # จาก (32, 32, 3) --> (1, 32, 32, 3)
    img_batch = np.reshape(image_array, [1, 32, 32, 3])
    print(f" เตรียมรูปภาพ '{IMAGE_PATH}' เรียบร้อยแล้ว")
except FileNotFoundError:
    print(f" ไม่พบไฟล์รูปภาพที่: {IMAGE_PATH}")
    exit()


# --- 4. ทำนายผลด้วยโมเดล ---
prediction = model.predict(img_batch)
# ผลลัพธ์ที่ได้จะเป็น array ที่มี 2 ค่า คือ [sin, cos]
# เช่น [[-0.9999, 0.0123]]
sin_val = prediction[0][0]
cos_val = prediction[0][1]


# --- 5. ถอดรหัสค่า Sin/Cos กลับเป็นตัวเลข ---
# ใช้ฟังก์ชัน arctan2 เพื่อแปลงค่า sin, cos กลับเป็นมุมในหน่วยเรเดียน
value_rad = np.arctan2(sin_val, cos_val)

# แปลงจากเรเดียนให้อยู่ในช่วง 0-1 (ตามที่เข้ารหัสไว้ตอนเทรน)
value_normalized = (value_rad / (2 * math.pi)) % 1

# แปลงค่ากลับเป็นสเกลเดิม (ตอนเทรนมีการหาร 10)
# เช่น 4.6 --> 0.46 ดังนั้นเราต้องคูณ 10 กลับ
final_value = value_normalized * 10


# --- 6. แสดงผลลัพธ์ ---
print("\n" + "="*30)
print(f"  ผลการทำนายค่าจากหน้าปัด")
print("="*30)
print(f"  ค่าที่อ่านได้: {final_value:.2f}") # แสดงผลเป็นทศนิยม 2 ตำแหน่ง
print("="*30)
print(f"(ข้อมูลดิบ: sin={sin_val:.4f}, cos={cos_val:.4f})")