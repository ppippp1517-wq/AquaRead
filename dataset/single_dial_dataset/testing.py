import os
import pandas as pd
from ultralytics import YOLO

# โหลดโมเดล classify ที่เทรนแล้ว
model = YOLO("D:/projectCPE/dataset/runs/classify/cls_dial_reader3/weights/best.pt")


# path ภาพจริง และ label ที่รู้
test_path = "D:/projectCPE/dataset/test_real_images"
label_file = "D:/projectCPE/dataset/labels_real.txt"

df = pd.read_csv(label_file)

total = 0
correct = 0
errors = []

for _, row in df.iterrows():
    filename, true_label = row["filename"], str(row["label"])
    img_path = os.path.join(test_path, filename)

    if not os.path.exists(img_path):
        print(f" ไม่พบไฟล์ {filename}")
        continue

    result = model(img_path)
    pred = str(result[0].probs.top1)

    total += 1
    if pred == true_label:
        correct += 1
    else:
        errors.append((filename, true_label, pred))

# สรุปผล
accuracy = correct / total * 100
print(f"\n Accuracy on real-world images: {accuracy:.2f}% ({correct}/{total})")

if errors:
    print("\n Misclassified images:")
    for fname, true, pred in errors:
        print(f"  {fname}: true = {true}, predicted = {pred}")
