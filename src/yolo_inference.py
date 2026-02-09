import cv2
from ultralytics import YOLO
import easyocr

# โหลดโมเดล YOLOv8 ที่ฝึกแล้ว
model = YOLO('D:/projectCPE/dataset/yolov8n.pt')  # ระบุเส้นทางไปยังไฟล์ yolov8n.pt ของคุณ

# อ่านภาพจาก ESP32-CAM
image_path = 'D:/projectCPE/dataset/images/test/test1.jpg'  # กำหนดเส้นทางไปยังภาพที่ต้องการทดสอบ
image = cv2.imread(image_path)  # อ่านภาพจากไฟล์

# ตรวจจับวัตถุในภาพ (ตัวเลข)
results = model(image)  # ใช้โมเดล YOLOv8 เพื่อตรวจจับวัตถุในภาพ

# แสดงผลการตรวจจับ (bounding boxes)
results[0].plot()  # แสดงภาพที่มี bounding boxes รอบตัวเลขหรือวัตถุที่ตรวจพบ

# ดึงข้อมูล bounding boxes (x, y, width, height)
boxes = results[0].boxes.xywh  # ผลลัพธ์จาก YOLOv8 (xywh: x, y, w, h)

# ใช้ EasyOCR อ่านตัวเลขจาก bounding boxes
reader = easyocr.Reader(['en'])  # ใช้ภาษาอังกฤษสำหรับ OCR

# แสดง bounding boxes และตัดภาพจากกรอบที่ตรวจพบ
for box in boxes:
    x, y, w, h = box[:4]
    print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")  # แสดงข้อมูลของ bounding box

    # ตัดภาพจาก bounding box
    cropped_image = image[int(y):int(y+h), int(x):int(x+w)]  # ตัดภาพจากกรอบที่ตรวจพบ
    
    # แสดงภาพที่ตัด
    cv2.imshow("Cropped Image", cropped_image)  # แสดงภาพที่ตัด

    # ใช้ OCR อ่านตัวเลขจากภาพที่ตัด
    result = reader.readtext(cropped_image)  # ใช้ EasyOCR เพื่ออ่านค่าจากภาพที่ตัด
    for detection in result:
        print("Detected Text: ", detection[1])  # แสดงผลตัวเลขที่ OCR อ่านได้

    cv2.waitKey(0)  # รอจนกว่าจะกดปุ่มเพื่อแสดงภาพ

cv2.destroyAllWindows()  # ปิดหน้าต่างทั้งหมดหลังจากการแสดงผล
