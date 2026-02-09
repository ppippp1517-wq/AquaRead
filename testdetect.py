# 🔧 Meter Reading Script: YOLOv8 + TFLite Angle Detection (class 1–3) + OCR (class 0)
import os
import cv2
import numpy as np
import pandas as pd
import math
import re
import tensorflow as tf
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pytesseract

# 📌 Config Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model_path = 'D:/projectCPE/runs/detect/train/weights/best.pt'
tflite_model_path = 'D:/projectCPE/neural-network-analog-needle-readout-main/models/ana-cont/ana-cont_1601_s2.tflite'
test_images_path = 'D:/projectCPE/dataset/images/test'
detect_images_path = 'D:/projectCPE/dataset/images/detect_images'
output_path = 'D:/projectCPE/dataset/images/cropped_images'
csv_output_path = 'D:/projectCPE/dataset/result.csv'
debug_crop_path = 'D:/projectCPE/dataset/images/debug_crop'

# 📁 Prepare folders
os.makedirs(detect_images_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(debug_crop_path, exist_ok=True)

# 🚀 Load YOLO model
model = YOLO(model_path)

# 🔁 Load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 🔍 Predict angle + digit from needle image (class 1–3)
def predict_angle_and_digit(image_path, interpreter):
    try:
        img = Image.open(image_path).convert('L').resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 32, 32, 1)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        angle_deg = interpreter.get_tensor(output_details[0]['index'])[0][0]

        print(f"📐 Predicted angle: {angle_deg:.2f} deg")
        value = round((angle_deg / 360) * 10, 1)
        digit = int(((angle_deg + 18) % 360) // 36)
        return value, digit
    except Exception as e:
        print("TFLite error:", e)
        return None, None

# ✅ Load TFLite interpreter
interpreter = load_tflite_model(tflite_model_path)

# 🔍 Predict with YOLO and TFLite
results = model.predict(source=test_images_path, conf=0.25)
all_data = []

for result in results:
    image_path = result.path
    image_name = os.path.basename(image_path)
    print(f"\n→ กำลังประมวลผล: {image_name}")

    img_with_boxes = result.plot()
    cv2.imwrite(os.path.join(detect_images_path, f"detected_{image_name}"), img_with_boxes)

    df = result.to_df()
    if df.empty:
        print("  ⚠️ ไม่พบวัตถุในภาพนี้")
        continue

    original_img = Image.open(image_path)
    boxes = df['box'].values
    class_ids = df['class'].values
    class_names = df['name'].values

    detection_data = []
    for i in range(len(boxes)):
        detection_data.append({
            'box': boxes[i],
            'class_id': class_ids[i],
            'class_name': class_names[i]
        })

    # 📦 Filter target classes
    required_class_ids = [0, 1, 2, 3]
    ocr_result_by_class = {}
    filtered = [d for d in detection_data if d['class_id'] in required_class_ids]
    filtered_sorted = sorted(filtered, key=lambda x: x['class_id'])

    for i, det in enumerate(filtered_sorted):
        box = det['box']
        class_id = det['class_id']
        class_name = det['class_name']

        x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
        cropped = original_img.crop((x1, y1, x2, y2))
        enhanced = ImageEnhance.Contrast(cropped).enhance(2.0)

        crop_path = os.path.join(output_path, f"class{class_id}_{image_name}_{i+1}.png")
        debug_path = os.path.join(debug_crop_path, f"class{class_id}_{image_name}_{i+1}.png")
        enhanced.save(crop_path)
        enhanced.save(debug_path)  # 🔍 save debug crop

        if class_id == 0:
            # 🔤 OCR สำหรับ digital display (ใช้วิธีเดิม + resize เพิ่มความชัด)
            resized = enhanced.resize((enhanced.width * 2, enhanced.height * 2))
            text = pytesseract.image_to_string(resized, config='--psm 6')
            print(f"📄 OCR raw (class 0): '{text}'")
            cleaned_text = re.sub(r'\D', '', text.strip())
            if not cleaned_text:
                print("⚠️ ไม่พบตัวเลขหลัง OCR")
            ocr_result_by_class[class_id] = cleaned_text if cleaned_text else '0'
        elif class_id in [1, 2, 3]:
            # 📐 ใช้ TFLite ทำนายมุมสำหรับเข็ม (class 1–3)
            value, digit = predict_angle_and_digit(crop_path, interpreter)
            if digit is not None:
                ocr_result_by_class[class_id] = str(digit)
            else:
                ocr_result_by_class[class_id] = '0'
            print(f"   🔧 class {class_id} → angle_value: {value}, mapped_digit: {digit}")

    int_part = ocr_result_by_class.get(0, '0')
    decimal1 = ocr_result_by_class.get(1, '0')
    decimal2 = ocr_result_by_class.get(2, '0')
    decimal3 = ocr_result_by_class.get(3, '0')
    combined_number = f"{int_part}.{decimal1}{decimal2}{decimal3}"

    print(f"\n ค่าที่อ่านได้: {combined_number}")

    all_data.append({
        'image': image_name,
        'digital_x': int_part,
        'x001': decimal1,
        'x0001': decimal2,
        'x00001': decimal3,
        'total': combined_number
    })

# 💾 Save to CSV
df_csv = pd.DataFrame(all_data)
df_csv.to_csv(csv_output_path, index=False)
print(f"\n✅ บันทึกผลทั้งหมดลงไฟล์: {csv_output_path}")