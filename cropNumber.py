import cv2
import os

# === CONFIG ===
input_folder = 'D:/projectCPE/augmented_resized'  # โฟลเดอร์ภาพหน้าปัด
output_folder = 'D:/projectCPE/digit_crops'       # ที่เก็บภาพที่ crop แล้ว
os.makedirs(output_folder, exist_ok=True)

# === Mouse Event State ===
drawing = False
ix, iy = -1, -1
rois = []
img_display = None
img = None

def mouse_crop(event, x, y, flags, param):
    global ix, iy, drawing, rois, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(rois) >= 5:
            print("⚠️ ครอบได้ไม่เกิน 5 ช่องตัวเลข")
            return
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img.copy()
        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        rois.append((x1, y1, x2, y2))
        print(f"➕ เพิ่มกรอบ: {(x1, y1)} → {(x2, y2)}")
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

# === Loop through images ===
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]

for file in image_files:
    rois = []
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ โหลดไม่ได้: {file}")
        continue

    img_display = img.copy()
    window_name = "Draw ROI (สูงสุด 5 ช่อง, กด s=บันทึก, n=ข้าม, q=ออก)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)
    cv2.setMouseCallback(window_name, mouse_crop)

    while True:
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            print(f"⏭️ ข้าม: {file}")
            break

        elif key == ord('q'):
            print("🛑 จบการทำงาน")
            cv2.destroyAllWindows()
            exit()

        elif key == ord('s'):
            if len(rois) == 0:
                print("⚠️ ยังไม่ได้ครอบกรอบใดเลย")
                break

            save_dir = os.path.join(output_folder, os.path.splitext(file)[0])
            os.makedirs(save_dir, exist_ok=True)

            # เรียงกรอบจากซ้ายไปขวา (x1)
            rois_sorted = sorted(rois, key=lambda r: r[0])

            for idx, (x1, y1, x2, y2) in enumerate(rois_sorted):
                crop = img[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (28, 28))
                gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                out_path = os.path.join(save_dir, f'digit_{idx+1}.jpg')
                cv2.imwrite(out_path, gray)
                print(f"✅ บันทึก: {out_path}")
            break

cv2.destroyAllWindows()
