import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Config ===
IMAGE_FOLDER = r"D:\projectCPE\dataset\images\test"
ROTATION_ANGLE = -5.0  # หมุนทวนเข็มเป็นองศา

# === Step 1: Load all images in folder ===
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"❌ No images found in folder: {IMAGE_FOLDER}")
    exit()

print(f"📂 Found {len(image_files)} image(s): {image_files}")

# === Loop through each image ===
for filename in image_files:
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load: {filename}")
        continue

    # Convert color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    # --- Step 2: Detect circle with Hough ---
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)

    min_radius = int(min(width, height) * 0.3)
    max_radius = int(min(width, height) * 0.7)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=height // 4,
        param1=100,
        param2=60,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Default center and radius
    x_center, y_center, r = width // 2, height // 2, min(width, height) // 2

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best_circle = max(circles, key=lambda c: c[2])
        x_center, y_center, r = best_circle
        print(f"🎯 {filename}: Detected circle at ({x_center},{y_center}), r={r}")
    else:
        print(f"⚠️ {filename}: No circle detected, using image center fallback")

    # --- Step 3: Crop circle ROI ---
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x_center, y_center), r, 255, -1)
    cropped_circular = np.full_like(image_rgb, 255)
    cropped_circular = np.where(mask[..., None] == 255, image_rgb, cropped_circular)

    # Bounding box crop
    top_crop = max(y_center - r, 0)
    bottom_crop = min(y_center + r, height)
    left_crop = max(x_center - r, 0)
    right_crop = min(x_center + r, width)

    circular_roi = cropped_circular[top_crop:bottom_crop, left_crop:right_crop]
    mask_cropped = mask[top_crop:bottom_crop, left_crop:right_crop]

    # คำนวณ center ใน ROI
    local_center_x = float(x_center - left_crop)
    local_center_y = float(y_center - top_crop)

# หมุนภาพ
    rotation_matrix = cv2.getRotationMatrix2D((local_center_x, local_center_y), float(ROTATION_ANGLE), 1.0)
    rotated_roi = cv2.warpAffine(
    circular_roi,
    rotation_matrix,
    (circular_roi.shape[1], circular_roi.shape[0]),
    flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(255, 255, 255)
)


    mask_rotated = cv2.warpAffine(
        mask_cropped.astype(np.uint8),
        rotation_matrix,
        (mask_cropped.shape[1], mask_cropped.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    rotated_roi = np.where(mask_rotated[..., None] == 255, rotated_roi, 255)

    # --- Step 5: Save rotated images ---
    output_jpg = os.path.join(IMAGE_FOLDER, f"rotated_{filename}")
    cv2.imwrite(output_jpg, cv2.cvtColor(rotated_roi, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved rotated image: {output_jpg}")

    # Save transparent PNG
    alpha = np.where(mask_rotated == 255, 255, 0).astype(np.uint8)
    rotated_bgr = cv2.cvtColor(rotated_roi, cv2.COLOR_RGB2BGR)
    rgba_final = cv2.merge([rotated_bgr[:, :, 0], rotated_bgr[:, :, 1], rotated_bgr[:, :, 2], alpha])
    output_png = os.path.join(IMAGE_FOLDER, f"rotated_{os.path.splitext(filename)[0]}.png")
    cv2.imwrite(output_png, rgba_final)
    print(f"✅ Saved transparent PNG: {output_png}")

print("🎉 All images processed!")
