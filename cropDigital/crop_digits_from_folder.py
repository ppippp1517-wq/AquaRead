
import os
import cv2
import json
import glob
import numpy as np

# ===================== CONFIG =====================
REFERENCE_PATH = r"D:\projectCPE\cropDigital\Reference.jpg"  # <-- เปลี่ยนพาธได้
INPUT_DIR      = r"D:\projectCPE\dataset\images\capture_images"
OUTPUT_DIR     = r"D:\projectCPE\dataset\images\cropdigital_images"
N_SLOTS        = 6             # จำนวนช่องตัวเลขในหน้าต่าง
MATCH_THRESH   = 0.5           # ค่าความคล้าย template (0..1) ต่ำกว่านี้จะลอง ORB
SAVE_CONFIG    = True          # บันทึกไฟล์ roi_config.json สำหรับอ้างอิงภายหลัง
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_digit_window_and_rois(ref_bgr, n_slots=6):
    """หาเฟรมหน้าต่างตัวเลข + กำหนด ROIs ในภาพอ้างอิง (อิงจากเฮอริสติก)"""
    H, W = ref_bgr.shape[:2]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(ref_gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        aspect = w/(h+1e-6)
        rel_area = area/(W*H)
        if 0.005 < rel_area < 0.2 and 2.0 < aspect < 8.0 and y < H*0.6:
            candidates.append((area, x,y,w,h))
    if not candidates:
        # fallback: กรอบที่กว้างที่สุดในครึ่งบน
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if y < int(H*0.6):
                candidates.append((w*h, x,y,w,h))

    if not candidates:
        raise RuntimeError("ไม่พบกรอบหน้าต่างตัวเลขอัตโนมัติ ลองเปลี่ยนภาพอ้างอิงหรือปรับ heuristics")

    candidates.sort(reverse=True, key=lambda t: t[0])
    _, x,y,w,h = candidates[0]

    # ขยับเข้าเล็กน้อย หลีกเลี่ยงขอบหนา
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.08)
    win_x = max(0, x + pad_x)
    win_y = max(0, y + pad_y)
    win_w = min(W - win_x, w - 2*pad_x)
    win_h = min(H - win_y, h - 2*pad_y)

    # แบ่งช่องเท่ากันภายในหน้าต่าง
    gap = int(win_w * 0.01)
    slot_w = (win_w - 2*gap) // n_slots
    slot_h = win_h - 2*gap
    slot_y = win_y + gap

    rois = []
    for i in range(n_slots):
        sx = win_x + gap + i * slot_w
        shrink = int(slot_w * 0.06)  # หดหลบขอบแนวตั้ง
        sx += shrink
        sw = slot_w - 2*shrink
        rois.append((sx, slot_y, sw, slot_h))

    # แพตช์จัดแนว (สี่เหลี่ยมเล็ก ๆ ซ้าย/ขวาบนภายในกรอบหน้าต่าง)
    patch_size = int(min(win_w, win_h) * 0.12)
    pl_x = max(win_x + int(win_w*0.04), 0)
    pl_y = max(win_y + int(win_h*0.12), 0)
    pr_x = min(win_x + win_w - patch_size - int(win_w*0.04), W - patch_size)
    pr_y = pl_y

    patch_left_box  = (pl_x, pl_y, patch_size, patch_size)
    patch_right_box = (pr_x, pr_y, patch_size, patch_size)

    return (win_x, win_y, win_w, win_h), rois, patch_left_box, patch_right_box

def crop_box(img, box):
    x,y,w,h = box
    return img[y:y+h, x:x+w]

def estimate_affine_by_templates(img_gray, ref_shape, tpl_left, tpl_right, ref_left_xy, ref_right_xy):
    """หาค่าแปลงเชิงเส้น (Affine) จากการแมตช์แพตช์ซ้าย/ขวา"""
    H, W = ref_shape[:2]
    tL_gray = cv2.cvtColor(tpl_left,  cv2.COLOR_BGR2GRAY) if tpl_left.ndim==3 else tpl_left
    tR_gray = cv2.cvtColor(tpl_right, cv2.COLOR_BGR2GRAY) if tpl_right.ndim==3 else tpl_right

    res1 = cv2.matchTemplate(img_gray, tL_gray, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img_gray, tR_gray, cv2.TM_CCOEFF_NORMED)
    _, max1, _, loc1 = cv2.minMaxLoc(res1)
    _, max2, _, loc2 = cv2.minMaxLoc(res2)

    # top-left ของตำแหน่งพบแพตช์ในภาพใหม่
    P1 = np.float32([loc1[0], loc1[1]])
    P2 = np.float32([loc2[0], loc2[1]])
    # จุดอ้างอิงในภาพ ref (ตำแหน่งที่แพตช์ควรอยู่)
    Q1 = np.float32(ref_left_xy)
    Q2 = np.float32(ref_right_xy)

    M, inliers = cv2.estimateAffinePartial2D(np.vstack([P1,P2]).reshape(-1,1,2),
                                            np.vstack([Q1,Q2]).reshape(-1,1,2),
                                            method=cv2.LMEDS)
    return M, max1, max2

def warp_orb(src_bgr, ref_bgr):
    """สำรอง: ใช้ ORB หาโฮโมกราฟีแล้ว warp ให้เท่ากับ ref"""
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return None
    matches = sorted(matches, key=lambda x: x.distance)[:200]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ref_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    Hmat, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
    if Hmat is None:
        return None
    Hh, Ww = ref_bgr.shape[:2]
    warped = cv2.warpPerspective(src_bgr, Hmat, (Ww, Hh))
    return warped

def main():
    ref_bgr = cv2.imread(REFERENCE_PATH)
    if ref_bgr is None:
        raise FileNotFoundError(f"ไม่พบภาพอ้างอิง: {REFERENCE_PATH}")
    H, W = ref_bgr.shape[:2]

    # --- เตรียมกรอบและแพตช์จาก reference ---
    digit_window, rois, patch_left_box, patch_right_box = detect_digit_window_and_rois(ref_bgr, N_SLOTS)

    # ตัดแพตช์จริงออกมาจากภาพอ้างอิง
    tpl_left  = crop_box(ref_bgr, patch_left_box)
    tpl_right = crop_box(ref_bgr, patch_right_box)
    # จุดอ้างอิง (จุดมุมบนซ้ายของแพตช์ในภาพอ้างอิง)
    ref_left_xy  = (patch_left_box[0],  patch_left_box[1])
    ref_right_xy = (patch_right_box[0], patch_right_box[1])

    # บันทึกคอนฟิกเป็นสัดส่วน
    if SAVE_CONFIG:
        def to_rel(box):
            x,y,w,h = box
            return dict(x=x/W, y=y/H, w=w/W, h=h/H)
        cfg = {
            "reference_image": REFERENCE_PATH,
            "image_size": {"W": W, "H": H},
            "digit_window": to_rel(digit_window),
            "num_slots": N_SLOTS,
            "rois": [to_rel(b) for b in rois],
            "alignment_patches": {
                "left":  to_rel(patch_left_box),
                "right": to_rel(patch_right_box)
            }
        }
        with open(os.path.join(OUTPUT_DIR, "roi_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # --- ประมวลผลทุกไฟล์ภาพ ---
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(INPUT_DIR, e))
    files = sorted(files)
    if not files:
        print("ไม่พบไฟล์ภาพใน", INPUT_DIR)
        return

    print(f"[INFO] Reference: {REFERENCE_PATH}")
    print(f"[INFO] Found {len(files)} image(s) in {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, fp in enumerate(files, 1):
        img = cv2.imread(fp)
        if img is None:
            print(f"[WARN] อ่านรูปไม่ได้: {fp}")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        M, scoreL, scoreR = estimate_affine_by_templates(
            img_gray, ref_bgr.shape, tpl_left, tpl_right, ref_left_xy, ref_right_xy
        )

        if M is not None and scoreL >= MATCH_THRESH and scoreR >= MATCH_THRESH:
            warped = cv2.warpAffine(img, M, (W, H))
            used = "Affine/Template"
        else:
            warped = warp_orb(img, ref_bgr)
            used = "ORB/Homography"

        if warped is None:
            print(f"[FAIL] จัดแนวไม่สำเร็จ: {os.path.basename(fp)}")
            continue

        # ครอปตาม ROIs ในพิกัดภาพอ้างอิง
        stem = os.path.splitext(os.path.basename(fp))[0]
        for i, (x,y,w,h) in enumerate(rois):
            crop = warped[y:y+h, x:x+w]
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_slot{i}.png")
            cv2.imwrite(out_path, crop)

        print(f"[OK] {idx}/{len(files)} {os.path.basename(fp)} -> {used} -> saved crops")

    print("[DONE] ครอปครบแล้ว ->", OUTPUT_DIR)

if __name__ == "__main__":
    main()
