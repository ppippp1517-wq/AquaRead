from ultralytics import YOLO

# โหลดโมเดล
model = YOLO(r"D:\projectCPE\runs\detect\train\weights\best.pt")

# ตรวจจับภาพใน test
results = model.predict(
    source="D:/projectCPE/dataset/images/test",
    conf=0.25,
    save=True
)

# แสดงผล
for r in results:
    print(f" Image: {r.path}")
    for box in r.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        print(f"Class {cls_id} | Confidence: {conf:.2f}")
