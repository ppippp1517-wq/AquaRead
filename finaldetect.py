#ดีเทคและครอบเลขดิจิทัลพร้อมรีไซส์
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ===== PATHS =====
model_path          = r'D:/projectCPE/dataset_digital/runs/detect/digital_det2/weights/best.pt' 
test_images_path    = r'D:/projectCPE/dataset/images/test'
crop_out_dir        = r'D:/projectCPE/dataset/images/cropdigital_images'   # ครอปเต็ม
resize_out_dir      = r'D:/projectCPE/dataset/images/resize_crop'          # ภาพย่อ 20x32

# ต้องการรูป 20x32 (กว้าง x สูง ตามที่แสดงในภาพตัวอย่าง)
RESIZE_WH = (20, 32)   # (width, height)

os.makedirs(crop_out_dir, exist_ok=True)
os.makedirs(resize_out_dir, exist_ok=True)

# ===== MODEL =====
model = YOLO(model_path)

def save_crops(src_path, boxes_xyxy, stem):
    """เซฟทุกกล่อง: ครอปเต็มลง crop_out_dir และเวอร์ชันย่อ 20x32 ลง resize_out_dir"""
    pil = Image.open(src_path).convert('RGB')

    # เรียงซ้าย -> ขวา เพื่อคงลำดับหลัก
    order = np.argsort(boxes_xyxy[:, 0])
    boxes_xyxy = boxes_xyxy[order]

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy, start=1):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        crop = pil.crop((x1, y1, x2, y2))

        # ครอปเต็ม
        crop.save(os.path.join(crop_out_dir, f"{stem}_{i}_roi.png"))

        # ภาพย่อ 20x32 (ตามภาพตัวอย่าง 20x32)
        crop.resize(RESIZE_WH, Image.BILINEAR)\
            .save(os.path.join(resize_out_dir, f"{stem}_{i}_20x32.png"))

# ===== RUN =====
for name in sorted(os.listdir(test_images_path)):
    if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        continue

    src = os.path.join(test_images_path, name)
    stem = os.path.splitext(name)[0]
    print(f"[PRED] {name}")

    rs = model.predict(
        source=src,
        imgsz=832,          # เพิ่มถ้าตัวเลขเล็กมาก เช่น 960
        conf=0.05,          # ลดเพื่อให้ครบทุกหลัก
        iou=0.5,
        agnostic_nms=False,
        verbose=False
    )
    r = rs[0]
    if r.boxes is None or len(r.boxes) == 0:
        print("   ❌ ไม่พบกล่องใดๆ"); continue

    boxes = r.boxes.xyxy.cpu().numpy()
    print(f"   ✓ พบ {len(boxes)} กล่อง → ครอปและย่อ")
    save_crops(src, boxes, stem)

print("เสร็จ ✓ ครอปเต็ม -> cropdigital_images และย่อ 20x32 -> resize_crop")
