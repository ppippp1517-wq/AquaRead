import os
import pandas as pd

# === PATH ===
image_folder = "D:/projectCPE/dataset/images/cropped_images"
csv_output = "D:/projectCPE/dataset/images/value_true.csv"

data = []

# === วนลูปไฟล์ภาพ ===
for fname in os.listdir(image_folder):
    if fname.endswith(".png"):
        parts = fname.split("_")
        class_part = parts[0]  # เช่น class3
        class_id = int(class_part.replace("class", ""))
        data.append({
            "filename": fname,
            "class_id": class_id,
            "value_true": ""  # ให้คุณใส่ค่ามุมจริงภายหลัง
        })

# === สร้างและบันทึก CSV ===
df = pd.DataFrame(data)
df.to_csv(csv_output, index=False)
print(f"สร้างไฟล์ CSV เรียบร้อย: {csv_output}")
