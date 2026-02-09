import cv2
import numpy as np
import os

def align_folder_images(reference_path, input_folder, output_folder, good_match_percent=0.2):
    os.makedirs(output_folder, exist_ok=True)

    # Load reference image
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise FileNotFoundError(f"❌ ไม่พบ reference image: {reference_path}")
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=8000)
    kp_ref, desc_ref = sift.detectAndCompute(ref_gray, None)
    print(f"🔹 Reference keypoints: {len(kp_ref)}")

    # Prepare matcher (ไม่ใช้ crossCheck)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Collect image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("❌ No images found in input folder.")
        return []

    results = []

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        test_img = cv2.imread(img_path)
        if test_img is None:
            print(f"⚠️ Failed to read {filename}, skipped.")
            continue

        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        kp_test, desc_test = sift.detectAndCompute(test_gray, None)

        # Skip if keypoints not enough
        if desc_test is None or len(kp_test) < 10:
            print(f"⚠️ Not enough keypoints in {filename}, skipped.")
            continue

        # 1️⃣ Match with Ratio Test
        knn_matches = bf.knnMatch(desc_ref, desc_test, k=2)
        good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

        print(f"🔹 {filename}: {len(good_matches)} good matches (after ratio test)")

        if len(good_matches) < 4:
            print(f"❌ Not enough matches for homography: {filename}")
            continue

        # 2️⃣ Compute Homography
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_test = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 3.0)

        # 3️⃣ Check Homography validity
        if H is None or np.isnan(H).any():
            print(f"❌ Homography invalid for {filename}, skipped.")
            continue

        # 4️⃣ Warp image
        h, w = ref_img.shape[:2]
        aligned_img = cv2.warpPerspective(test_img, H, (w, h))

        # 5️⃣ Compute Quality Score
        inliers = np.sum(mask) if mask is not None else 0
        quality_score = inliers / len(good_matches)

        if quality_score < 0.4:
            print(f"⚠️ {filename} skipped due to low quality ({quality_score:.2f})")
            continue

        # 6️⃣ Save aligned image
        output_path = os.path.join(output_folder, f"aligned_{filename}")
        cv2.imwrite(output_path, aligned_img)
        print(f"✅ {filename}: Quality Score = {quality_score:.2f}")

        # 7️⃣ Store result
        results.append({
            'filename': filename,
            'matches': len(good_matches),
            'quality': float(f"{quality_score:.3f}")
        })

    print("🎉 Alignment complete!")
    return results
