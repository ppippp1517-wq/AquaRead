import os
from PIL import Image
import pandas as pd
from ultralytics import YOLO

# === CONFIG ===
model_path = "D:/projectCPE/runs/detect/train/weights/best.pt"
input_images_path = "D:/projectCPE/dataset_new/images_new/train"
output_crop_path = "D:/projectCPE/dataset/images/cropped_images"
csv_output_path = "D:/projectCPE/dataset/images/dial_labels.csv"

os.makedirs(output_crop_path, exist_ok=True)

# โหลดโมเดล YOLO
model = YOLO(model_path)

# ตรวจจับ dial
results = model.predict(source=input_images_path, conf=0.25)

crop_data = []

for result in results:
    image_path = result.path
    image_name = os.path.basename(image_path)
    original_img = Image.open(image_path)

    df = result.to_df()
    if df.empty:
        continue

    for i, row in df.iterrows():
        class_id = int(row['class'])
        box = row['box']
        x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])

        crop_img = original_img.crop((x1, y1, x2, y2))

        crop_filename = f"class{class_id}_{image_name}_{i+1}.png"
        crop_img.save(os.path.join(output_crop_path, crop_filename))

        crop_data.append({
            'filename': crop_filename,
            'class_id': class_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })

# บันทึก CSV
df = pd.DataFrame(crop_data)
df.to_csv(csv_output_path, index=False)
print(f"\n ครอบภาพจาก train แล้ว: {len(df)} ภาพ")
print(f" เก็บไว้ที่: {output_crop_path}")
print(f"CSV labels: {csv_output_path}")
