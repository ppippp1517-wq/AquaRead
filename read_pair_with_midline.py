import os, re, glob, csv
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ===== CONFIG =====
MODEL_PAIR = r"D:/projectCPE/pair_ab_keras.h5"        # โมเดลคู่เลข a->b
IMG_INPUT  = r"D:/projectCPE/dataset/images/test"     # ใส่ได้ทั้ง "ไฟล์" หรือ "โฟลเดอร์"

W, H = 20, 32             # ขนาดอินพุตของโมเดล (width, height)
MID_RATIO = 0.50          # เส้นแบ่ง (0.50 = กลางภาพ, 0.45/0.55 ก็ได้)
AREA_THR  = 0.50          # ถ้าสัดส่วน "มวลความมืด" ใต้เส้น >= ค่านี้ ⇒ ตัดเป็นเลขถัดไป
IGNORE_X_MARGIN = 0.10    # ตัดขอบซ้าย/ขวาออก (สัดส่วนกว้าง) เพื่อลดผลจากกรอบ/เงา
USE_CLAHE = True          # เพิ่มคอนทราสต์ก่อนแปลงเป็น mask
MORPH_OPEN = True         # เปิดรูเล็ก ๆ เพื่อลด noise

# บันทึกผล
SAVE_CSV   = r"D:/projectCPE/midline_results.csv"
SAVE_PLOTS = True
PLOT_DIR   = r"D:/projectCPE/plots_midline"
PLOT_DPI   = 150
SHOW_PLOT  = True         # โชว์กราฟบนจอ
SHOW_FIRST_N = 1          # แสดงกราฟกี่รูปแรก
SAVE_PLOTS = True
PLOT_DIR   = r"D:/projectCPE/plots_midline"
PLOT_DPI   = 150           # อยากคมกว่านี้เพิ่มได้ เช่น 200–300
SHOW_PLOT  = False         # ถ้าแค่เซฟ ไม่ต้องโชว์บนจอ ให้ False จะรันเร็วกว่า

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

# ===== utils =====
def natural_key(p):
    b = os.path.basename(p)
    nums = re.findall(r'\d+', b)
    return [int(n) for n in nums] if nums else [b.lower()]

def collect_paths(inp):
    if os.path.isdir(inp):
        files=[]
        for ext in IMG_EXTS: files += glob.glob(os.path.join(inp, f"*{ext}"))
        return sorted(set(files), key=natural_key)
    elif os.path.isfile(inp):
        return [inp]
    else:
        raise FileNotFoundError(f"ไม่พบไฟล์/โฟลเดอร์: {inp}")

def pair_predict(model, roi_gray):
    # roi_gray: [H,W] uint8
    x = (roi_gray.astype(np.float32)/255.0)[None, ..., None]  # [1,H,W,1]
    p = model.predict(x, verbose=0)[0]                         # [10]
    a = int(p.argmax()); b=(a+1)%10; conf=float(p[a])
    return a,b,conf

def to_dark_mask(roi_gray):
    g = roi_gray.copy()
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        g = clahe.apply(g)
    # อินเวิร์ตให้ "มืด = 1"
    _, mask = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if MORPH_OPEN:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(np.float32)  # [H,W], 0/1

def dark_fraction_below(mask, mid_ratio=0.5, ignore_margin=0.1):
    H, W = mask.shape
    y_mid = int(round(mid_ratio*(H-1)))
    x0 = int(round(ignore_margin*W))
    x1 = W - x0
    if x1 <= x0: x0, x1 = 0, W
    roi = mask[:, x0:x1]
    tot = roi.sum() + 1e-6
    below = roi[y_mid+1:, :].sum()
    frac = float(below / tot)  # 0..1
    return frac, y_mid, (x0, x1)

def plot_one(img_rgb, roi_gray, a,b,conf, frac, y_mid, x_margin, out_path=None, show=False):
    H, W = roi_gray.shape
    fig = plt.figure(figsize=(9,5))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img_rgb); ax1.axis("off")
    ax1.set_title(f"{os.path.basename(out_path or 'preview')}\n{a}->{b} conf={conf:.2f}")
    # เส้นแบ่ง
    ax1.axhline(y_mid, color='C0', lw=3)

    # ขวา: แสดงสัดส่วน dark ต่อแถว + เส้นแบ่ง
    ax2 = fig.add_subplot(1,2,2)
    dark = (roi_gray.max() - roi_gray).astype(np.float32)
    row_dark = dark.sum(axis=1)
    y = np.arange(H)
    ax2.plot(row_dark, y, label="Row darkness (proxy)")
    ax2.invert_yaxis()
    ax2.axhline(y_mid, color='C0', lw=2, ls='-')
    ax2.set_xlabel("Darkness"); ax2.set_ylabel("Y")
    ax2.set_title(f"area_below={frac:.3f} | d1_disc={'%d'%(b if frac>=AREA_THR else a)}")
    ax2.legend()

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        print("  -> saved plot:", out_path)
    if show: plt.show()
    else: plt.close(fig)

# ===== main =====
def main():
    files = collect_paths(IMG_INPUT)
    if not files:
        print("ไม่พบรูปในโฟลเดอร์ที่ระบุ"); return

    pair_model = tf.keras.models.load_model(MODEL_PAIR, compile=False)

    rows=[["path","file","a","b","conf_pair","area_below","mid_ratio","area_thr","d1_disc"]]
    shown = 0

    for fp in files:
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            print(f"[WARN] อ่านรูปไม่ได้: {fp}"); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)

        a,b,conf = pair_predict(pair_model, roi_gray)
        mask = to_dark_mask(roi_gray)
        frac, y_mid, (x0,x1) = dark_fraction_below(mask, MID_RATIO, IGNORE_X_MARGIN)

        d1_disc = b if frac >= AREA_THR else a

        print(os.path.basename(fp),
              f"| pair {a}->{b} (conf={conf:.2f})",
              f"| area_below={frac:.3f}",
              f"| mid={MID_RATIO:.2f} thr={AREA_THR:.2f} -> d1_disc={d1_disc}")

        rows.append([fp, os.path.basename(fp), a,b,
                     f"{conf:.4f}", f"{frac:.6f}",
                     f"{MID_RATIO:.2f}", f"{AREA_THR:.2f}", d1_disc])

        if SAVE_PLOTS or (SHOW_PLOT and shown < SHOW_FIRST_N):
            out = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(fp))[0] + "_midline.png") if SAVE_PLOTS else None
            plot_one(img_rgb, roi_gray, a,b,conf, frac, y_mid, (x0,x1), out_path=out,
                     show=(SHOW_PLOT and shown < SHOW_FIRST_N))
            shown += 1

    if SAVE_CSV:
        os.makedirs(os.path.dirname(SAVE_CSV) or ".", exist_ok=True)
        with open(SAVE_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("Wrote CSV ->", SAVE_CSV, "rows:", len(rows)-1)

if __name__ == "__main__":
    main()
