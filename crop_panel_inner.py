# crop_panel_inner.py
import cv2, numpy as np

def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def _canny_edges(g):
    g = cv2.GaussianBlur(g, (3,3), 0)
    v = cv2.Canny(g, 60, 180, L2gradient=True)
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    return v

def _first_strong_run(proj, from_start=True, thr_ratio=0.25, min_run_ratio=0.04):
    """หา index แรกที่มีขอบ 'หนา' ต่อเนื่อง (ใช้กับบน/ล่าง/ซ้าย/ขวา)"""
    n = proj.shape[0]
    thr = max(3.0, float(proj.max()) * thr_ratio)
    min_run = max(4, int(round(n * min_run_ratio)))

    idx_range = range(0, n) if from_start else range(n-1, -1, -1)
    run, found = 0, None
    for i in idx_range:
        if proj[i] >= thr:
            run += 1
            if run >= min_run:
                found = i if from_start else i
                break
        else:
            run = 0
    if found is None:
        found = 0 if from_start else n-1
    return found

def _fallback_contour_rect(bw):
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w = bw.shape[:2]; return (0,0,w,h)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)
    x1,y1 = box[:,0].min(), box[:,1].min()
    x2,y2 = box[:,0].max(), box[:,1].max()
    return (x1,y1,x2,y2)

def crop_panel_inner(bgr_or_gray, inset_px=2, thr_ratio=0.25, min_run_ratio=0.04):
    """
    ตัด 'กรอบใน' ของหน้าปัด (ก่อน split 5 ช่อง)
    return: cropped_gray, (x1,y1,x2,y2), debug_bgr
    """
    g  = _to_gray(bgr_or_gray)
    ed = _canny_edges(g)

    # โปรไฟล์แนวตั้ง/แนวนอนของขอบ
    vproj = ed.sum(axis=0).astype(np.float32)  # ต่อคอลัมน์
    hproj = ed.sum(axis=1).astype(np.float32)  # ต่อแถว

    # ซ้าย/ขวา/บน/ล่าง จาก run ของขอบหนา
    x1 = _first_strong_run(vproj, from_start=True,  thr_ratio=thr_ratio, min_run_ratio=min_run_ratio)
    x2 = _first_strong_run(vproj, from_start=False, thr_ratio=thr_ratio, min_run_ratio=min_run_ratio)
    y1 = _first_strong_run(hproj, from_start=True,  thr_ratio=thr_ratio, min_run_ratio=min_run_ratio)
    y2 = _first_strong_run(hproj, from_start=False, thr_ratio=thr_ratio, min_run_ratio=min_run_ratio)

    # ตรวจ sanity: ถ้ากลับด้าน/แน่นเกินไป → fallback
    h, w = g.shape[:2]
    if (x2 - x1) < w * 0.4 or (y2 - y1) < h * 0.3:
        # ทำ binary คร่าว ๆ เพื่อหากรอบใหญ่สุด
        bw = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,7)
        x1,y1,x2,y2 = _fallback_contour_rect(bw)

    # ขยับเข้า (เลี่ยงเส้นดำ)
    x1 = max(0, x1 + inset_px);   y1 = max(0, y1 + inset_px)
    x2 = min(w, x2 - inset_px);   y2 = min(h, y2 - inset_px)
    if x2 <= x1: x1, x2 = 0, w
    if y2 <= y1: y1, y2 = 0, h

    cropped = g[y1:y2, x1:x2].copy()

    # debug overlay
    dbg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
    return cropped, (x1,y1,x2,y2), dbg
