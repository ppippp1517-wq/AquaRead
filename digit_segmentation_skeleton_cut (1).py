#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digit strip segmentation (N digits; default 5) using:
1) CLAHE + adaptive binarization
2) Deskew (PCA on foreground points)
3) Skeleton -> endpoints/junctions as seeds
4) Vertical projection minima + snap-to-grid
5) Crop each digit (tight on y) and pad to square
6) Strong frame suppression for edge columns

Author: ChatGPT (patched)
Requirements: Python 3.8+, OpenCV (cv2), NumPy
Optional: scikit-image (skeletonize), opencv-contrib (ximgproc.thinning)
"""

import os
import cv2
import numpy as np
import argparse

resize_target = None  # set in main()

# --- Optional: try scikit-image for skeletonization; otherwise fallback thinning ---
try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# --- Optional: ximgproc.thinning (fast + clean) ---
try:
    import cv2.ximgproc as xip
    _HAS_XIMGPROC = True
except Exception:
    _HAS_XIMGPROC = False


# -------------------- Utils --------------------
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p)


def clahe_grayscale(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray, block_size=21, C=10, gauss_ksize=3):
    """
    Adaptive (Gaussian) threshold + median denoise; fallback to Otsu if needed.
    Output: digits=white(255), background=black(0)
    """
    k = max(3, gauss_ksize | 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    try:
        bs = block_size if block_size % 2 == 1 else block_size + 1
        th = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs, C
        )
        th = cv2.medianBlur(th, 3)
    except Exception:
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def deskew(gray_or_bin, force_landscape=True):
    """
    Deskew ด้วย PCA ของจุด foreground เพื่อลดผลรบกวนจากขอบ/เส้น
    Return: rotated_gray, angle_deg
    """
    if len(gray_or_bin.shape) == 3:
        gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bin if gray_or_bin.dtype == np.uint8 else (gray_or_bin * 255).astype(np.uint8)

    bin_img = binarize(gray)

    ys, xs = np.nonzero(bin_img)
    if len(xs) < 10:
        out = gray
        if force_landscape and out.shape[0] > out.shape[1]:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        return out, 0.0

    coords = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=np.array([]))
    vx, vy = eigenvectors[0]
    angle = np.degrees(np.arctan2(vy, vx))

    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if force_landscape and rotated.shape[0] > rotated.shape[1]:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

    return rotated, angle


# -------------------- Skeleton & Seeds --------------------
def do_skeleton(binary):
    """
    binary expected: digits=white(255), bg=black(0)
    Returns a uint8 skeleton image (0/255).
    """
    if _HAS_XIMGPROC:
        return xip.thinning(binary)
    if _HAS_SKIMAGE:
        sk = skeletonize((binary > 0)).astype(np.uint8) * 255
        return sk
    prev = np.zeros_like(binary)
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0 or np.array_equal(img, prev):
            break
        prev = img.copy()
    return skel


def find_junction_and_endpoints(skel):
    sk = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    deg = neigh * sk

    ys, xs = np.where((sk == 1) & (deg == 1))
    endpoints = np.stack([xs, ys], axis=1) if len(xs) > 0 else np.empty((0, 2), dtype=int)

    yj, xj = np.where((sk == 1) & (deg >= 3))
    junctions = np.stack([xj, yj], axis=1) if len(xj) > 0 else np.empty((0, 2), dtype=int)
    return endpoints, junctions


# -------------------- Projection & Cuts --------------------
def vertical_projection(binary, band=(0.30, 0.70)):
    H, W = binary.shape
    y0 = int(max(0, min(H-1, H * band[0])))
    y1 = int(max(y0+1, min(H,   H * band[1])))

    colsum = (binary[y0:y1, :] > 0).sum(axis=0).astype(np.float32)
    if colsum.max() > 0:
        colsum = colsum / colsum.max()
    k = 9
    smoothed = np.convolve(colsum, np.ones(k) / k, mode='same')
    return smoothed


def pick_cut_positions(profile, seeds=None, num_digits=5, min_gap_ratio=0.12, W=None):
    if W is None:
        W = len(profile)
    need = num_digits - 1
    min_gap = max(6, int(W * min_gap_ratio))

    candidates = []
    for x in range(1, W - 1):
        if profile[x] <= profile[x - 1] and profile[x] <= profile[x + 1]:
            candidates.append((x, profile[x]))
    candidates.sort(key=lambda t: t[1])

    def far_enough(x, arr, mind):
        return all(abs(x - y) >= mind for y in arr)

    selected = []

    if seeds:
        seed_sorted = sorted(seeds)
        for sx in seed_sorted:
            nearest = None
            bestd = 1e9
            for x, _ in candidates:
                d = abs(x - sx)
                if d < bestd:
                    bestd = d
                    nearest = x
            if nearest is not None and 0 < nearest < W - 1 and far_enough(nearest, selected, min_gap):
                selected.append(int(nearest))
                if len(selected) == need:
                    break

    for x, _ in candidates:
        if len(selected) == need:
            break
        if 0 < x < W - 1 and far_enough(x, selected, min_gap):
            selected.append(int(x))

    if len(selected) < need:
        step = W / num_digits
        for i in range(1, num_digits):
            x = int(round(i * step))
            if 0 < x < W - 1 and far_enough(x, selected, min_gap):
                selected.append(x)
            if len(selected) == need:
                break

    selected = sorted(set([int(x) for x in selected if 0 < x < W - 1]))
    if len(selected) != need:
        step = W / num_digits
        selected = [int(round(i * step)) for i in range(1, num_digits)]
    return sorted(selected)


# -------------------- Frame/Noise helpers --------------------
def remove_frame_lines(digit_bin, v_ratio=0.60, h_ratio=0.60, edge_ratio=0.22):
    """ลบเส้นกรอบยาวใกล้ขอบภาพ (แนวตั้ง/แนวนอน) แล้วคืน binary ใหม่"""
    binu = digit_bin.copy()
    h, w = binu.shape

    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(v_ratio * h))))
    vert = cv2.morphologyEx(binu, cv2.MORPH_OPEN, kv)
    mask_v = np.zeros_like(binu)
    m = max(1, int(edge_ratio * w))
    mask_v[:, :m] = vert[:, :m]
    mask_v[:, w - m:] = vert[:, w - m:]
    binu = cv2.subtract(binu, mask_v)

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(h_ratio * w)), 1))
    hori = cv2.morphologyEx(binu, cv2.MORPH_OPEN, kh)
    mask_h = np.zeros_like(binu)
    mY = max(1, int(edge_ratio * h))
    mask_h[:mY, :] = hori[:mY, :]
    mask_h[h - mY:, :] = hori[h - mY:, :]
    binu = cv2.subtract(binu, mask_h)

    return binu


def tight_crop_from_binary(gray, bin_img, margin=2):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return gray, bin_img
    h, w = bin_img.shape
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    x1 = max(0, x1 - margin); x2 = min(w, x2 + margin)
    y1 = max(0, y1 - margin); y2 = min(h, y2 + margin)
    return gray[y1:y2, x1:x2], bin_img[y1:y2, x1:x2]


def shave_borders_by_colfill(gray, bin_img, max_frac=0.40, thr=0.52, hyst=0.44):
    H, W = bin_img.shape
    col = (bin_img > 0).sum(axis=0) / max(1, H)

    L = 0
    while L < int(max_frac * W) and col[L] > thr:
        L += 1
    while L > 0 and col[L-1] > hyst:
        L -= 1

    R = W - 1
    while R > W-1-int(max_frac * W) and col[R] > thr:
        R -= 1
    while R < W-1 and col[R+1] > hyst:
        R += 1

    L = max(0, min(L, W-2)); R = max(L+1, min(R, W-1))
    return gray[:, L:R+1], bin_img[:, L:R+1]


def strip_edge_bands(bin_img, edge_frac=0.18, fill_thresh=0.55):
    H, W = bin_img.shape
    colfill = (bin_img > 0).sum(axis=0) / max(1, H)
    rowfill = (bin_img > 0).sum(axis=1) / max(1, W)

    L, R = 0, W - 1
    while L < int(W*edge_frac) and colfill[L] > fill_thresh: L += 1
    while R > W-1-int(W*edge_frac) and colfill[R] > fill_thresh: R -= 1

    T, B = 0, H - 1
    while T < int(H*edge_frac) and rowfill[T] > fill_thresh: T += 1
    while B > H-1-int(H*edge_frac) and rowfill[B] > fill_thresh: B -= 1

    L = max(0, min(L, W-2)); R = max(L+1, min(R, W-1))
    T = max(0, min(T, H-2)); B = max(T+1, min(B, H-1))
    return bin_img[T:B+1, L:R+1]


def filter_components_central(gray, bin_img, min_area_ratio=0.02):
    H, W = bin_img.shape
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_img, 8)
    if num <= 1:
        return gray, bin_img
    areas = stats[1:, cv2.CC_STAT_AREA]
    cx0, cy0 = W/2.0, H/2.0

    keep = np.zeros_like(bin_img)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        cx, cy = cents[i]
        central = (abs(cx-cx0) <= 0.35*W) and (abs(cy-cy0) <= 0.45*H)
        if a >= max(12, min_area_ratio*H*W) or central:
            keep[labels==i] = 255

    if cv2.countNonZero(keep) == 0 and len(areas):
        i = 1 + int(np.argmax(areas))
        keep[labels==i] = 255

    ys, xs = np.where(keep>0)
    x1, x2 = xs.min(), xs.max()+1; y1, y2 = ys.min(), ys.max()+1
    return gray[y1:y2, x1:x2], keep[y1:y2, x1:x2]


def recenter_by_moments(gray, bin_img):
    H, W = bin_img.shape
    M = cv2.moments((bin_img>0).astype(np.uint8))
    if M["m00"] == 0: return gray, bin_img
    cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
    tx = int(round(W/2 - cx)); ty = int(round(H/2 - cy))
    Mshift = np.float32([[1,0,tx],[0,1,ty]])
    g = cv2.warpAffine(gray, Mshift, (W,H), flags=cv2.INTER_LINEAR, borderValue=255)
    b = cv2.warpAffine(bin_img, Mshift, (W,H), flags=cv2.INTER_NEAREST, borderValue=0)
    return g, b


def keep_main_component(gray, bin_img, min_area_ratio=0.01, border=1):
    H, W = bin_img.shape
    if bin_img.dtype != np.uint8:
        bin_img = (bin_img > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    keep = np.zeros_like(bin_img)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if x <= border or y <= border or x + w >= W - border or y + h >= H - border:
            continue
        if area < max(10, int(min_area_ratio * H * W)):
            continue
        keep[labels == i] = 255
    if cv2.countNonZero(keep) == 0:
        if num > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[labels == i] = 255
        else:
            keep = bin_img.copy()

    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return gray, keep
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return gray[y1:y2, x1:x2], keep[y1:y2, x1:x2]


def pad_square_and_resize(gray, bin_img, size=None, bg=255):
    h, w = gray.shape[:2]
    s = max(h, w)
    top = (s - h) // 2; bottom = s - h - top
    left = (s - w) // 2; right = s - w - left
    g = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=bg)
    b = cv2.copyMakeBorder(bin_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    if size:
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (size, size), interpolation=cv2.INTER_NEAREST)
    return g, b


def hard_shave_side(bin_img, side='left', max_ratio=0.45, fill_thr=0.60):
    H, W = bin_img.shape
    colfill = (bin_img > 0).sum(axis=0) / max(1, H)
    if side == 'left':
        limit = min(int(max_ratio * W), W-2)
        k = 0
        while k < limit and colfill[k] > fill_thr:
            k += 1
        bin_img[:, :k] = 0
    else:
        limit = min(int(max_ratio * W), W-2)
        k = W-1
        while k > W-1-limit and colfill[k] > fill_thr:
            k -= 1
        bin_img[:, k+1:] = 0
    return bin_img


def clear_white_touching_border(bin_img):
    """
    ลบคอมโพเนนต์ 'ขาว' ที่แตะขอบทั้งสี่ด้านออกด้วย floodFill (digits=white, bg=black)
    """
    img = bin_img.copy()
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    for x in range(w):
        if img[0, x] != 0:
            cv2.floodFill(img, mask, (x, 0), 0)
        if img[h-1, x] != 0:
            cv2.floodFill(img, mask, (x, h-1), 0)
    mask[:] = 0
    for y in range(h):
        if img[y, 0] != 0:
            cv2.floodFill(img, mask, (0, y), 0)
        if img[y, w-1] != 0:
            cv2.floodFill(img, mask, (w-1, y), 0)
    return img


def kill_edge_borders(bin_img, max_cut=0.35, thr=0.50, smooth=9):
    b = bin_img.copy()
    H, W = b.shape
    colfill = (b > 0).sum(axis=0) / max(1, H)
    k = max(1, smooth)
    colf = cv2.blur(colfill.reshape(1, -1).astype(np.float32), (1, k)).ravel()

    L, Llim = 0, int(max_cut * W)
    while L < Llim and colf[L] > thr:
        L += 1
    R, Rlim = W - 1, int(max_cut * W)
    while R > W - 1 - Rlim and colf[R] > thr:
        R -= 1

    if L > 0: b[:, :L] = 0
    if R < W - 1: b[:, R + 1:] = 0
    return b


def remove_border_tall_sticks(bin_img, side_band=0.30, min_h=0.90, max_w=0.30):
    H, W = bin_img.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    keep = np.zeros_like(bin_img)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        cx = x + w/2
        touches = (x == 0) or (x + w >= W-1) or (y == 0) or (y + h >= H-1)
        near_side = (cx < side_band*W) or (cx > (1.0 - side_band)*W)
        if touches and near_side and h >= min_h*H and w <= max_w*W:
            continue  # drop frame pillar
        keep[labels == i] = 255
    return keep


def suppress_vertical_near_edges(bin_img, rel_len=0.93, band_frac=0.36):
    H, W = bin_img.shape
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(rel_len*H))))
    vlong = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kv)
    mask = np.zeros_like(bin_img)
    m = int(band_frac*W)
    mask[:, :m] = 255; mask[:, W-m:] = 255
    vlong = cv2.bitwise_and(vlong, mask)
    return cv2.subtract(bin_img, vlong)


def suppress_internal_horizontal(bin_img, rel_len=0.65):
    h, w = bin_img.shape
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(rel_len * w)), 1))
    bars = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kh)
    return cv2.subtract(bin_img, bars)


def trim_caps_by_rowfill(gray, bin_img, top_frac=0.45, bot_frac=0.34, thr=0.48, hyst=0.42):
    H, W = bin_img.shape
    rowfill = (bin_img > 0).sum(axis=1) / max(1, W)

    T = 0
    while T < int(H*top_frac) and rowfill[T] > thr: T += 1
    while T < H-2 and rowfill[T] > hyst: T += 1

    B = H-1
    while B > H-1-int(H*bot_frac) and rowfill[B] > thr: B -= 1
    while B > 1 and rowfill[B] > hyst: B -= 1

    T = max(0, min(T, H-2)); B = max(T+1, min(B, H-1))
    return gray[T:B+1, :], bin_img[T:B+1, :]


def remove_side_pillars(bin_img, band_frac=0.35, min_h=0.65, max_w=0.45):
    H, W = bin_img.shape
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num <= 1:
        return bin_img

    keep = np.zeros_like(bin_img)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        is_side = (cx < band_frac * W) or (cx > (1.0 - band_frac) * W)
        is_tall = (h >= min_h * H)
        not_too_wide = (w <= max_w * W)
        if is_side and is_tall and not_too_wide:
            continue
        keep[labels == i] = 255
    return keep


def keep_thick_strokes(bin_img, dt_frac=0.40, dilate_iter=1):
    """
    เก็บเฉพาะส่วนที่หนาของลายเส้นด้วย Distance Transform
    ใช้กับ binary แบบ digits=white(255), bg=black(0)
    """
    if bin_img.dtype != np.uint8:
        bin_img = (bin_img > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(bin_img)  # bg=255, fg=0
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    m = float(dt.max())
    if m <= 0:
        return bin_img.copy()
    thick = (dt > (dt_frac * m)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if dilate_iter > 0:
        thick = cv2.dilate(thick, k, iterations=dilate_iter)
    return thick


def remove_lines_hough(gray, bin_img, min_len_ratio=0.75, max_gap=2):
    """ลบเส้นตรงยาวแนวตั้ง/แนวนอนด้วย HoughLinesP บนภาพหลัก"""
    H, W = bin_img.shape
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50,
        minLineLength=int(min_len_ratio*min(H, W)),
        maxLineGap=max_gap
    )
    out = bin_img.copy()
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(x2-x1) <= 2 or abs(y2-y1) <= 2:      # เกือบตั้ง/นอน
                cv2.line(out, (x1,y1), (x2,y2), 0, 3)
    return out


# -------------------- Digit cleanup (patched) --------------------
def post_trim_digit(digit_gray, digit_bin, margin=2):
    """
    ลำดับใหม่: เคลียร์ของแตะขอบ → ลบกรอบยาว/เสายาวแถบขอบ → ตัด band บน/ล่าง →
    เลือกคอมโพเนนต์กลาง → เก็บเส้นหนา (ตัวเลข) → tight + กันขอบ → Hough
    """
    g = digit_gray.copy()
    b = digit_bin.copy()

    # 0) ลบคอมโพเนนต์ที่แตะขอบก่อน
    b = clear_white_touching_border(b)

    # 1) ลบเส้นกรอบยาว + โกนขอบหนามาก
    b = remove_frame_lines(b, v_ratio=0.95, h_ratio=0.85, edge_ratio=0.55)
    b = kill_edge_borders(b, max_cut=0.35, thr=0.50, smooth=9)

    # 2) เสาตั้งยาวในแถบซ้าย/ขวา
    b = suppress_vertical_near_edges(b, rel_len=0.93, band_frac=0.36)
    b = remove_border_tall_sticks(b, side_band=0.34, min_h=0.90, max_w=0.30)

    # 3) ตัดแถบบน/ล่างที่ยังหนา
    g, b = trim_caps_by_rowfill(g, b, top_frac=0.45, bot_frac=0.34, thr=0.48, hyst=0.42)

    # 4) เลือกคอมโพเนนต์ตัวเลขก่อน (ใหญ่/กลาง)
    g, b = filter_components_central(g, b, min_area_ratio=0.015)

    # 5) เก็บเฉพาะเส้นหนาของตัวเลข เพื่อตัดกรอบบางๆ ออก
    b = keep_thick_strokes(b, dt_frac=0.40, dilate_iter=1)

    # 6) tight-crop + pad กันขอบ
    g, b = tight_crop_from_binary(g, b, margin=1)
    g = cv2.copyMakeBorder(g, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
    b = cv2.copyMakeBorder(b, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)

    # 7) ลบเส้นตรงยาวตกค้าง (อ่อน)
    b = remove_lines_hough(g, b, min_len_ratio=0.75, max_gap=2)

    return g, b


# -------------------- Crop Cleanup ภายในแต่ละหลัก (ลบเส้นรบกวน/กรอบ/เสา)--------------------
def crop_digits(original_gray, binary, cuts, pad=4):
    H, W = binary.shape
    xs = [0] + cuts + [W]
    crops, boxes = [], []

    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        x1e = max(0, x1 - pad)
        x2e = min(W, x2 + pad)

        roi_bin  = binary[:, x1e:x2e]
        roi_gray = original_gray[:, x1e:x2e]

        rowsum = (roi_bin > 0).sum(axis=1)
        ys = np.where(rowsum > 0)[0]
        y1, y2 = (int(ys[0]), int(ys[-1] + 1)) if ys.size else (0, roi_bin.shape[0])

        digit_gray = roi_gray[y1:y2, :]
        if digit_gray.size == 0:
            continue  # safety

        # ใช้ binarize ที่อ่อนลงเพื่อไม่สร้างกรอบปลอม
        digit_bin  = binarize(digit_gray, block_size=31, C=8, gauss_ksize=3)

        # ล้างกรอบ + คอมโพเนนต์กลาง + เน้นเส้นหนา
        digit_gray, digit_bin = post_trim_digit(digit_gray, digit_bin, margin=2)

        # เก็บคอมโพเนนต์หลักอีกครั้ง (เผื่อ Hough ลบส่วนเกิน)
        digit_gray, digit_bin = keep_main_component(digit_gray, digit_bin, min_area_ratio=0.01, border=1)

        # pad/resize
        digit_gray_sq, digit_bin_sq = pad_square_and_resize(digit_gray, digit_bin, size=resize_target)

        crops.append((digit_gray_sq, digit_bin_sq))
        boxes.append((x1e, y1, x2e, y2))

    return crops, boxes


# -------------------- Pipeline --------------------
def segment_digits(image_bgr, num_digits=5, min_gap_ratio=0.12,
                   block_size=21, C=10, gauss_ksize=3):
    # 1) preprocess
    gray0 = clahe_grayscale(image_bgr)
    gray, ang = deskew(gray0)
    bin_img = binarize(gray, block_size=block_size, C=C, gauss_ksize=gauss_ksize)

    # 2) skeleton + seeds
    sk = do_skeleton(bin_img)
    endpoints, junctions = find_junction_and_endpoints(sk)

    H, W = sk.shape
    seed_x = []
    if len(junctions) > 0:
        seed_x.extend([int(x) for x, y in junctions])
    if len(endpoints) > 0:
        seed_x.extend([int(x) for x, y in endpoints if 0.2 * H < y < 0.8 * H])
    seed_x = sorted(list({x for x in seed_x if 0 <= x < W}))

    # 3) vertical projection (แถบกลาง) + จำกัดช่วงใช้งานแนวนอน
    prof_full = vertical_projection(bin_img, band=(0.30, 0.70))

    th = 0.05
    active_idx = np.where(prof_full > th)[0]
    if len(active_idx) > 0:
        L, R = int(active_idx[0]), int(active_idx[-1])
        if R - L < num_digits:
            L, R = 0, len(prof_full) - 1
    else:
        L, R = 0, len(prof_full) - 1

    sub_prof = prof_full[L:R+1]
    Wsub = len(sub_prof)
    step = max(1.0, Wsub / float(num_digits))
    win  = int(step * 0.35)

    minima = [x for x in range(1, Wsub-1)
              if sub_prof[x] <= sub_prof[x-1] and sub_prof[x] <= sub_prof[x+1]]

    cuts_sub = []
    for i in range(1, num_digits):
        target = int(round(i * step))
        lo, hi = max(1, target - win), min(Wsub - 2, target + win)
        cands = [x for x in minima if lo <= x <= hi]
        if cands:
            xbest = min(cands, key=lambda x: sub_prof[x])
        else:
            xbest = target
        cuts_sub.append(int(xbest))

    min_gap = max(6, int(Wsub * min_gap_ratio))
    cuts_sub = sorted(set(cuts_sub))
    fixed = []
    for x in cuts_sub:
        if not fixed or (x - fixed[-1]) >= min_gap:
            fixed.append(x)
    cuts_sub = fixed

    if len(cuts_sub) != (num_digits - 1):
        cuts_sub = [int(round(i * step)) for i in range(1, num_digits)]

    cuts = [x + L for x in cuts_sub]

    # 4) crop
    crops, boxes = crop_digits(gray, bin_img, cuts)

    out = {
        "gray": gray, "binary": bin_img, "skeleton": sk,
        "cuts": cuts, "boxes": boxes, "endpoints": endpoints, "junctions": junctions,
        "angle_deg": ang, "profile": prof_full
    }
    return crops, out


def visualize_debug(base_img_bgr, meta, scale=1.0):
    vis = base_img_bgr.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    sk = meta["skeleton"]
    cuts = meta["cuts"]
    endpoints = meta["endpoints"]
    junctions = meta["junctions"]

    overlay = vis.copy()
    overlay[sk > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)

    H, W = vis.shape[:2]
    for x in cuts:
        cv2.line(vis, (x, 0), (x, H - 1), (0, 0, 255), 2)

    for (x, y) in junctions:
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), -1)
    for (x, y) in endpoints:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return vis


def save_crops(crops, outdir, prefix="digit", save_bin=False):
    ensure_dir(outdir)
    paths = []
    for i, (g, b) in enumerate(crops):
        pg = os.path.join(outdir, f"{prefix}_{i}.png")
        cv2.imwrite(pg, g)  # grayscale
        paths.append(pg)
        if save_bin:
            pb = os.path.join(outdir, f"{prefix}_{i}_bin.png")
            cv2.imwrite(pb, b)
    return paths


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="D:/projectCPE/class0.png", help="Path to ROI image (digit strip).")
    ap.add_argument("--outdir", default="D:/projectCPE/out_digits", help="Output directory for crops.")
    ap.add_argument("--num-digits", type=int, default=5, help="Number of digits in the strip.")
    ap.add_argument("--min-gap-ratio", type=float, default=0.12,
                    help="Minimal horizontal gap between cuts relative to ROI width.")
    ap.add_argument("--debug", action="store_true", help="Save debug overlay and intermediates.")
    ap.add_argument("--save-bin", action="store_true", help="Also save binary crops for each digit.")
    # binarization params (global / for segment)
    ap.add_argument("--block", type=int, default=21, help="Adaptive threshold block size (odd).")
    ap.add_argument("--C", type=int, default=10, help="Adaptive threshold constant C.")
    ap.add_argument("--gauss-ksize", type=int, default=3, help="Gaussian blur kernel size (odd).")
    # output size
    ap.add_argument("--resize", type=int, default=64, help="Resize output digits to NxN (0=keep size).")
    return ap.parse_args()


def main():
    args = parse_args()

    global resize_target
    resize_target = args.resize if args.resize and args.resize > 0 else None

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    crops, meta = segment_digits(
        img,
        num_digits=args.num_digits,
        min_gap_ratio=args.min_gap_ratio,
        block_size=args.block,
        C=args.C,
        gauss_ksize=args.gauss_ksize
    )
    paths = save_crops(crops, args.outdir, save_bin=args.save_bin)

    print("Saved digit crops:")
    for p in paths:
        print("  ", p)

    if args.debug:
        ensure_dir(args.outdir)
        dbg = visualize_debug(img, meta, scale=1.0)
        cv2.imwrite(os.path.join(args.outdir, "_debug_overlay.png"), dbg)
        cv2.imwrite(os.path.join(args.outdir, "_binary.png"), meta["binary"])
        cv2.imwrite(os.path.join(args.outdir, "_skeleton.png"), meta["skeleton"])
        cv2.imwrite(os.path.join(args.outdir, "_gray_deskewed.png"), meta["gray"])
        print("Saved debug images to:", args.outdir)
        print(f"Deskew angle (deg): {meta['angle_deg']:.3f}")


if __name__ == "__main__":
    main()
