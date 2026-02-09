import os
import pandas as pd
import re

# CONFIG
crop_dir = "D:/projectCPE/dataset/images/crop"
csv_output_path = "D:/projectCPE/dataset/images/angle_labels.csv"

data = []

for fname in os.listdir(crop_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("ตรวจสอบ:", fname)

        # ดักชื่อไฟล์แบบ classX_imgY.jpg_Z.png
        match = re.search(r'\.jpg_(\d+)\.png$', fname, re.IGNORECASE)
        if match:
            angle = int(match.group(1))
            data.append({'filename': fname, 'angle': angle})
        else:
            print(f"❌ ไม่พบ angle ในไฟล์: {fname}")

# ถ้าไม่มีข้อมูล ให้สร้าง DataFrame ว่างพร้อมคอลัมน์ เพื่อป้องกัน error
if data:
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame(columns=['filename', 'angle'])

# SAVE
df.to_csv(csv_output_path, index=False)
print(f"\n✅ สร้างไฟล์ angle_labels.csv แล้ว: {csv_output_path}")
print(f"🧾 จำนวนภาพที่บันทึก: {len(df)}")
