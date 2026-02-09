import albumentations as A
import cv2
import os

# กำหนดเส้นทางโฟลเดอร์ภาพ
input_folder = "D:/projectCPE/dataset/imagesAll"
output_folder = "D:/projectCPE/dataset/imagesFull"

os.makedirs(output_folder, exist_ok=True)

# กำหนดการเปลี่ยนแปลงที่ต้องการทำ
transform = A.Compose([
    A.RandomRotate90(p=0.5),  # หมุน 90 องศา
    A.HorizontalFlip(p=0.5),   # พลิกแนวนอน
    A.RandomBrightnessContrast(p=0.5),  # ปรับความสว่างและคอนทราสต์
    A.Blur(blur_limit=3, p=0.3),  # เบลอภาพเล็กน้อย
])

# โหลดภาพจากโฟลเดอร์และทำการ Augment
for image_name in os.listdir(input_folder):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)
        
        # ทำ Augmentation
        augmented_image = transform(image=img)['image']
        
        # บันทึกภาพที่ถูก Augment
        augmented_image_path = os.path.join(output_folder, f"aug_{image_name}")
        cv2.imwrite(augmented_image_path, augmented_image)

print("Data Augmentation เสร็จสิ้น")
