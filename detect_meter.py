import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import pytesseract
import re
import cv2

# ตั้งค่าตัวแปร tesseract_cmd ให้ชี้ไปที่ `tesseract.exe` ที่ติดตั้งในเครื่อง
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # ตั้งตามที่ติดตั้ง Tesseract

# เส้นทางที่ใช้สำหรับโมเดลที่เทรนและโฟลเดอร์ที่ต้องการตรวจจับ
model_path = 'D:/projectCPE/dataset/runs/detect/train12/weights/best.pt'  # ใช้โมเดลที่ฝึกแล้ว
test_images_path = 'D:/projectCPE/dataset/images/test'  # โฟลเดอร์ที่มีภาพที่ต้องการตรวจจับ
detect_images_path = 'D:/projectCPE/dataset/images/detect_images'  # โฟลเดอร์ที่จะบันทึกภาพที่ตรวจจับแล้ว
output_path = 'D:/projectCPE/dataset/images/cropped_images'  # โฟลเดอร์ที่จะบันทึกภาพที่ตัดออกมา

# สร้างโฟลเดอร์สำหรับบันทึกภาพที่ตรวจจับและภาพที่ตัดออกมา (ถ้าไม่มี)
if not os.path.exists(detect_images_path):
    os.makedirs(detect_images_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# โหลดโมเดล YOLOv8 ที่เทรนแล้ว
model = YOLO(model_path)  # โหลดโมเดลที่ฝึกแล้ว

# ตรวจจับในภาพจากโฟลเดอร์ที่กำหนด
results = model.predict(source=test_images_path, conf=0.25) # ใช้ไฟล์จากโฟลเดอร์ test

img_with_boxes = results[0].plot()
cv2.imwrite("result_with_boxes.jpg", img_with_boxes)
print("บันทึกภาพที่มีกรอบไว้ที่ result_with_boxes.jpg แล้ว")


# ใช้ to_df() เพื่อแปลงข้อมูลผลลัพธ์มาเป็น DataFrame จากแต่ละ Result object
df = results[0].to_df()  # ดึงข้อมูลจากผลลัพธ์ที่ 1 (แรก) เป็น DataFrame

# ตรวจสอบว่า dataframe มีข้อมูลหรือไม่
if df.empty:
    print("ไม่พบวัตถุในภาพ")
else:
    # ดึงข้อมูล bounding boxes, confidence, และ class labels
    boxes = df['box'].values  # ข้อมูล bounding boxes ที่เป็น dictionary
    confidences = df['confidence'].values  # ค่า confidence ของการตรวจจับ
    labels = df['name'].values  # ชื่อของวัตถุที่ตรวจจับได้

    # คำนวณค่าตำแหน่งกลาง (center) และขนาด (width, height) จากค่า bounding boxes
    calculated_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        calculated_boxes.append([x_center, y_center, width, height])

    # แปลง calculated_boxes ให้เป็น numpy array
    calculated_boxes = np.array(calculated_boxes)

    # จัดเรียง bounding boxes ตามตำแหน่ง X (ซ้ายไปขวา)
    sorted_boxes = sorted(zip(calculated_boxes, confidences), key=lambda x: x[0][0])

    # ดึงข้อมูลจาก sorted_boxes
    sorted_positions = [box[0] for box in sorted_boxes]
    sorted_confidences = [box[1] for box in sorted_boxes]

    # ตัดภาพตามตำแหน่งที่ระบุและบันทึกภาพที่ตัดออกมา
    cropped_images = []
    for i, box in enumerate(sorted_positions):
        x1, y1, x2, y2 = map(int, box)  # แปลงเป็น integer
        
        # ตรวจสอบค่าพิกัดเพื่อไม่ให้เกิดการข้ามกัน
        if x1 > x2:  # ถ้า x1 มากกว่า x2 ให้สลับค่า
            x1, x2 = x2, x1
        if y1 > y2:  # ถ้า y1 มากกว่า y2 ให้สลับค่า
            y1, y2 = y2, y1

        # วนลูปผ่านทุกไฟล์ในโฟลเดอร์ test
        for image_file in os.listdir(test_images_path):
            if image_file.endswith(('.jpg', '.png', '.jpeg')):  # ตรวจสอบว่าเป็นไฟล์ภาพ
                image_path = os.path.join(test_images_path, image_file)  # สร้างเส้นทางไฟล์
                print(f"กำลังเปิดไฟล์: {image_path}")

                # ใช้ PIL เปิดไฟล์ภาพ
                img = Image.open(image_path)

                # ตรวจจับและบันทึกภาพที่ตรวจจับแล้วใน detect_images
                img.save(f"{detect_images_path}/detected_{image_file}")

                # ตัดภาพจาก bounding box
                cropped_image = img.crop((x1, y1, x2, y2))  # ตัดภาพตาม bounding box
                cropped_images.append(cropped_image)

                # ใช้เทคนิคการปรับภาพให้คมชัดขึ้น (contrast) เพื่อช่วยให้ OCR ทำงานได้ดีขึ้น
                enhancer = ImageEnhance.Contrast(cropped_image)
                enhanced_image = enhancer.enhance(2.0)  # ปรับความคมชัด (ปรับค่าตามที่เหมาะสม)

                # บันทึกภาพที่ตัดออกมา
                enhanced_image.save(f"{output_path}/cropped_image_{i+1}.png")  # บันทึกภาพที่ตัดออกมา

    # ใช้ OCR บนภาพที่ตัดออกมา
    text_numbers = []
    for cropped_image in cropped_images:
        text = pytesseract.image_to_string(cropped_image, config='--psm 6')  # PSM 6 ใช้ในการตรวจจับบรรทัดเดียว
        text_numbers.append(text.strip())  # เก็บตัวเลขที่ตรวจจับได้

    # ทำความสะอาดผลลัพธ์ OCR เพื่อให้ได้ตัวเลขที่ถูกต้อง
    cleaned_numbers = []
    for text in text_numbers:
        # กรองเฉพาะตัวเลขที่ถูกต้อง
        cleaned_text = re.sub(r'\D', '', text)  # ลบทุกอย่างที่ไม่ใช่ตัวเลข
        cleaned_numbers.append(cleaned_text)

    # รวมตัวเลขที่ตรวจจับได้เป็นผลลัพธ์เดียว (เช่น 19.5482 m3)
    final_result = '.'.join(cleaned_numbers)
    print("ตัวเลขที่ตรวจจับได้:", final_result)
