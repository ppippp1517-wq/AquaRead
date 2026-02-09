import cv2
import os
import csv

IMAGE_FOLDER = "D:/projectCPE/annotate_raw"
OUTPUT_CSV = "D:/projectCPE/dataset/labels/keypoints_annotated_normalized.csv"
RESIZE_WIDTH = 300

image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
current_index = 0
keypoints_data = []
clicks = []

def click_event(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f" Clicked: ({x}, {y})")
        if len(clicks) == 2:
            img_copy = param.copy()
            cv2.circle(img_copy, clicks[0], 4, (0, 255, 0), -1)
            cv2.circle(img_copy, clicks[1], 4, (0, 0, 255), -1)
            cv2.line(img_copy, clicks[0], clicks[1], (255, 0, 0), 2)
            cv2.imshow("Annotate", img_copy)

while current_index < len(image_files):
    filename = image_files[current_index]
    img_path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f" ไม่สามารถเปิดภาพได้: {filename}")
        current_index += 1
        continue

    clicks = []
    height, width = img.shape[:2]
    img_resized = cv2.resize(img, (RESIZE_WIDTH, int(height * RESIZE_WIDTH / width)))
    scale_x = width / RESIZE_WIDTH
    scale_y = height / int(height * RESIZE_WIDTH / width)

    cv2.imshow("Annotate", img_resized)
    cv2.setMouseCallback("Annotate", click_event, img_resized)
    print(f"\n📷 {filename} (คลิก center → tip แล้วกด s เพื่อเซฟ / n ข้าม / q ออก)")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(clicks) == 2:
            cx, cy = clicks[0]
            px, py = clicks[1]
            cx, cy = int(cx * scale_x), int(cy * scale_y)
            px, py = int(px * scale_x), int(py * scale_y)

            #  Normalize
            norm_cx = cx / width
            norm_cy = cy / height
            norm_px = px / width
            norm_py = py / height

            keypoints_data.append([filename, norm_cx, norm_cy, norm_px, norm_py])
            print(f" Saved: {filename}")
            break
        elif key == ord('n'):
            print(f" Skipped: {filename}")
            break
        elif key == ord('q'):
            print(" Quit")
            current_index = len(image_files)
            break

    current_index += 1

cv2.destroyAllWindows()

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "cx", "cy", "px", "py"])  # พิกัด normalized แล้ว
    writer.writerows(keypoints_data)

print(f"\n บันทึกพิกัด normalized แล้วที่: {OUTPUT_CSV}")
