# file: trim_strip_by_slots_v4.py
import cv2, numpy as np, os
from math import atan2, degrees

# ---------- Deskew ----------
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

# ---------- Binarize ----------
def _binarize(img_bgr, use_adaptive=True):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    if use_adaptive:
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)
    else:
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    return th

# ---------- Cluster x-center -> 5 กลุ่ม ----------
def _cluster_to_5(bboxes):
    xs = np.array([x + w/2 for (x,y,w,h) in bboxes], dtype=np.float32).reshape(-1,1)
    K = 5
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, _ = cv2.kmeans(xs, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    groups = [[] for _ in range(K)]
    for b, lb in zip(bboxes, labels.flatten()):
        groups[lb].append(b)
    merged = []
    for g in groups:
        if not g:  # กลุ่มว่าง ข้าม
            continue
        xs = [x for (x,_,_,_) in g]; ys = [y for (_,y,_,_) in g]
        x2s = [x+w for (x,_,w,_) in g]; y2s = [y+h for (_,y,_,h) in g]
        merged.append((min(xs), min(ys), max(x2s)-min(xs), max(y2s)-min(ys)))
    merged = sorted(merged, key=lambda b: b[0])
    if len(merged) > 5:
        best = None
        for i in range(len(merged)-4):
            span = (merged[i+4][0]+merged[i+4][2]) - merged[i][0]
            if best is None or span < best[0]:
                best = (span, merged[i:i+5])
        merged = best[1]
    return merged

# ---------- Projection fallback (แนวนอน) ----------
def projection_fallback(img_bgr, pad_ratio=0.06, min_width_ratio=0.70):
    """
    ใช้ vertical projection + smoothing + dilation หา x1-x2 ของแถบตัวเลข
    แล้วขยายซ้าย-ขวาเผื่อเลขที่จาง/ติดกรอบ
    """
    H0, W0 = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 1D smoothing on column sum ---
    col = th.sum(axis=0).astype(np.float32)
    k = max(3, W0 // 100 * 2 + 1)                 # kernel ~2% ของความกว้าง
    col_smooth = cv2.GaussianBlur(col.reshape(1,-1), (1, k), 0).ravel()

    # --- adaptive threshold + dilation (fill gaps) ---
    thr = 0.10 * (col_smooth.max() if col_smooth.max() > 0 else 1)
    mask = (col_smooth > thr).astype(np.uint8)
    # close gaps up to ~2% width
    dil = int(max(2, round(0.02 * W0)))
    mask = cv2.dilate(mask.reshape(1,-1), np.ones((1,dil), np.uint8), iterations=1).ravel()

    idx = np.where(mask > 0)[0]
    if len(idx) == 0:
        return img_bgr

    # เอา segment ต่อเนื่องที่ยาวที่สุด
    best = None; s = None
    for i in range(len(mask)):
        if mask[i] and s is None: s = i
        if (not mask[i] or i == len(mask)-1) and s is not None:
            e = i if not mask[i] else i
            if best is None or (e - s) > (best[1] - best[0]):
                best = (s, e)
            s = None

    x1, x2 = best
    # ขยายซ้าย-ขวาแบบสัดส่วน + แบบพิกเซลคงที่
    pad = int(pad_ratio * (x2 - x1 + 1))
    extra = max(6, int(0.015 * W0))               # เผื่ออย่างน้อย ~1.5% หรือ 6px
    x1 = max(0, x1 - pad - extra)
    x2 = min(W0-1, x2 + pad + extra)

    # ถ้ากว้างยังน้อยกว่าเกณฑ์ขั้นต่ำ ให้ขยายไปจนถึง min_width_ratio
    if (x2 - x1 + 1) < int(min_width_ratio * W0):
        need = int(min_width_ratio * W0) - (x2 - x1 + 1)
        left_add = need // 2
        right_add = need - left_add
        x1 = max(0, x1 - left_add)
        x2 = min(W0-1, x2 + right_add)

    # ตัดบนล่างด้วย row projection
    sub = th[:, x1:x2+1]
    row = sub.sum(axis=1).astype(np.float32)
    rk = max(3, H0 // 120 * 2 + 1)
    row = cv2.GaussianBlur(row.reshape(-1,1), (rk,1), 0).ravel()
    rthr = 0.10 * (row.max() if row.max() > 0 else 1)
    idy = np.where(row > rthr)[0]
    if len(idy) == 0:
        return img_bgr[:, x1:x2+1]

    y1 = max(0, int(idy[0] - 2))
    y2 = min(H0-1, int(idy[-1] + 2))
    return img_bgr[y1:y2+1, x1:x2+1]

# ---------- Trim หลัก ----------
def trim_by_5_slots(
    img_bgr,
    slot_ar_range=(0.20, 1.40),    # รูปทรงช่องเดี่ยว: กว้าง/สูง
    slot_area_ratio=(0.0015, 0.35),# สัดส่วนพื้นที่ช่องเดี่ยว เทียบทั้ง ROI
    pad_in_ratio=0.02,             # ตัดขอบกรอบหนาออกเล็กน้อย
    min_strip_ratio=0.60           # ถ้ากว้าง < 60% ของภาพเดิม → ใช้ projection ขยาย
):
    img = deskew_edge_pca(img_bgr)
    th = _binarize(img, use_adaptive=True)

    H, W = th.shape
    contours, _ = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if x<=1 or y<=1 or x+w>=W-1 or y+h>=H-1:
            continue
        area = w*h
        if not (slot_area_ratio[0]*W*H <= area <= slot_area_ratio[1]*W*H):
            continue
        ar = w/float(h)
        if not (slot_ar_range[0] <= ar <= slot_ar_range[1]):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)
        if len(approx) != 4:
            continue
        cand.append((x,y,w,h))

    # ≥3 ชิ้น: รวมกรอบ แล้วตรวจสอบความกว้าง
    if len(cand) >= 3:
        group5 = _cluster_to_5(cand)
        if group5:
            xs  = [x for (x,_,_,_) in group5]
            ys  = [y for (_,y,_,_) in group5]
            x2s = [x+w for (x,_,w,_) in group5]
            y2s = [y+h for (_,y,_,h) in group5]
            x1,y1,x2,y2 = min(xs), min(ys), max(x2s), max(y2s)

            # ตัด padding ด้านในเพื่อลดกรอบหนา
            pad = int(pad_in_ratio * min(x2-x1, y2-y1))
            x1 = max(0, x1+pad); y1 = max(0, y1+pad)
            x2 = min(W-1, x2-pad); y2 = min(H-1, y2-pad)

            crop = img[y1:y2, x1:x2]

            # ถ้ากว้างยังเล็กเกิน (เสี่ยงหายตัวซ้าย/ขวา) → ใช้ projection ขยาย
            if (x2 - x1) < min_strip_ratio * W:
                crop = projection_fallback(img)

            return crop

    # <3 ชิ้น: ใช้ projection ตรงๆ (เสถียรกว่า)
    return projection_fallback(img)

# ---------- Run ----------
if __name__ == "__main__":
    in_path  = r"D:\projectCPE\class1.png"
    out_path = r"D:\projectCPE\class1_trimmed_v4.png"

    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(in_path)

    trimmed = trim_by_5_slots(img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, trimmed)
    print("Saved:", out_path, trimmed.shape[1], "x", trimmed.shape[0])
