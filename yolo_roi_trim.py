# yolo_roi_trim.py
import cv2, numpy as np
from math import atan2, degrees

def deskew_edge_pca(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 80, 160)
    ys, xs = np.where(e > 0)
    if len(xs) < 60:
        return img_bgr
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, eig = cv2.PCACompute(pts, mean=None)
    vx, vy = eig[0]
    angle = degrees(atan2(vy, vx))
    M = cv2.getRotationMatrix2D((img_bgr.shape[1]/2, img_bgr.shape[0]/2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def trim_to_digit_window(roi_bgr):
    """ครอปให้เหลือเฉพาะแถบเลขดิจิทัล และตัดส่วนเกินขวาด้วย projection"""
    RIGHT_INNER_PAD = 0.03  # ตัดขอบกรอบหนาด้านใน (ซ้าย/ขวา/บน/ล่าง)
    AR_RANGE   = (2.5, 8.0)  # อัตราส่วนกว้าง/สูงของแถบ 5 ช่อง
    AREA_RANGE = (0.25, 0.95)

    img = deskew_edge_pca(roi_bgr)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    binv = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 8)
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(binv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binv.shape

    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x+w >= W-1 or y+h >= H-1:
            continue
        area = w*h
        if not (AREA_RANGE[0]*W*H <= area <= AREA_RANGE[1]*W*H):
            continue
        ar = w / float(h)
        if not (AR_RANGE[0] <= ar <= AR_RANGE[1]):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)
        if len(approx) != 4:
            continue
        if best is None or area > best[0]:
            best = (area, (x,y,w,h))

    if best is None:
        m = int(0.05 * min(H, W))
        crop = img[m:H-m, m:W-m]
        return crop

    _, (x,y,w,h) = best

    # ตัดขอบหนาด้านในเล็กน้อย
    pad = int(RIGHT_INNER_PAD * min(w, h))
    xL = max(0, x + pad)
    yT = max(0, y + pad)
    w2 = max(1, w - 2*pad)
    h2 = max(1, h - 2*pad)
    crop = img[yT:yT+h2, xL:xL+w2]

    # ---- ตัดส่วนเกินด้านขวาด้วย vertical projection (กันลูกศร) ----
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    col = th.sum(axis=0).astype(np.float32)
    # smooth + fill gaps เล็กน้อย
    k = max(3, (crop.shape[1] // 100) * 2 + 1)
    col = cv2.GaussianBlur(col.reshape(1,-1), (1, k), 0).ravel()
    mask = (col > 0.12 * (col.max() if col.max() > 0 else 1)).astype(np.uint8)
    gap = max(2, int(0.02 * crop.shape[1]))
    mask = cv2.dilate(mask.reshape(1,-1), np.ones((1, gap), np.uint8), iterations=1).ravel()

    idx = np.where(mask > 0)[0]
    if len(idx) > 0:
        x1, x2 = int(idx[0]), int(idx[-1])
        pad_proj = int(0.03 * (x2 - x1 + 1))
        x1 = max(0, x1 - pad_proj)
        x2 = min(crop.shape[1]-1, x2 + pad_proj)
        crop = crop[:, x1:x2+1]
        # ---- Force cut ขวา กันลูกศร/เกิน ----
        # ===== FINAL HARD RIGHT CLIP (kill arrow) =====
    MAX_RIGHT_RATIO = 0.86   # เก็บไว้ถึง 88% ซ้าย->ขวา (จูนได้ 0.86–0.90)
    MAX_RIGHT_PX    = 40  # หรือกำหนดเป็นพิกเซลคงที่ เช่น 40

    h, w = crop.shape[:2]

    if MAX_RIGHT_PX is not None:
        new_w = max(1, w - int(MAX_RIGHT_PX))
    else:
        new_w = max(1, int(MAX_RIGHT_RATIO * w))

    # อย่าตัดจนกินเลขกลาง ๆ: บังคับอย่างน้อย 70% ของกว้างเดิม
    new_w = max(new_w, int(0.70 * w))

    crop = crop[:, :new_w]



    return crop

# ==========================
#  RUN DIRECTLY FOR TESTING
# ==========================
if __name__ == "__main__":
    import sys, os

    # ตัวอย่าง: python yolo_roi_trim.py D:/projectCPE/class1.png D:/projectCPE/class1_trimmed.png
    if len(sys.argv) < 3:
        print("Usage: python yolo_roi_trim.py <input_path> <output_path>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    img = cv2.imread(in_path)
    if img is None:
        print("❌ ไม่พบไฟล์:", in_path)
        sys.exit(1)

    trimmed = trim_to_digit_window(img)
    cv2.imwrite(out_path, trimmed)
    print(f"✅ Saved trimmed ROI -> {out_path}")
