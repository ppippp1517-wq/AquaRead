#อ่านเลขเลื่อน อละเขยีนค่าพร้อมกราฟ กรอบเหลือง
import os, glob, cv2, csv
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ================== CONFIG ==================
MODEL_DIGIT = r"D:/projectCPE/Train_CNN_Digital-Readout_Version_5.0.0.h5"  # 0..9 + NaN(=10)
MODEL_PAIR  = r"D:/projectCPE/pair_ab_keras.h5"                            # a->b (0..9)
TEST_IMAGE_DIR = r"D:/projectCPE/dataset/images/test"                      # โฟลเดอร์ ROI

# ปรับภาพก่อนส่งเข้าโมเดลหลัก (เหมือนโค้ดเดิมของคุณ)
DO_CLAHE   = True
DO_UNSHARP = True
GAMMA      = 0.95         # <1 เข้มขึ้น, >1 สว่างขึ้น

# เกณฑ์ตัดสิน "เลขกำลังเลื่อน" แบบ midline-area
W, H = 20, 32             # อินพุตของโมเดล pair (width, height)
MID_RATIO = 0.50          # เส้นแบ่งแนวนอน (0..1), 0.50 = กลางภาพ
AREA_THR  = 0.60          # ถ้าสัดส่วนมวลความมืดใต้เส้น >= เกณฑ์ → เอาเลข b
IGNORE_X_MARGIN = 0.10    # ตัดขอบซ้าย/ขวาทิ้ง (ลดผลกรอบ/เงา)
MIN_PAIR_CONF = 0.00      # conf จากโมเดล pair ต่ำกว่าเกณฑ์นี้ จะ fallback เป็น NaN

# (ทางเลือก) เกณฑ์รายคู่เลข เช่น 6->7 เข้มกว่า
PAIR_THR = {
    # 6: 0.68,
    # 7: 0.62,
}

# เซฟผล/กราฟสำหรับเคส NaN
SAVE_CSV   = r"D:/projectCPE/transition_decisions.csv"
SAVE_PLOTS = True
PLOT_DIR   = r"D:/projectCPE/plots_transition"
PLOT_DPI   = 150
SHOW_PLOT  = False        # True = โชว์กราฟบนจอ (ช้าลง)

# --- Confidence thresholds ---
DIGIT_CONF_MIN = 0.80   # ถ้าโมเดลหลักมั่นใจ >= ค่านี้ ใช้ผลโมเดลหลักทันที
PAIR_CONF_MIN  = 0.50   # conf จากโมเดล pair ต้องมากกว่าค่านี้ด้วย
# ============================================

def enhance_digit(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert('L'))
    if DO_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
    if DO_UNSHARP:
        blur = cv2.GaussianBlur(gray, (0,0), 1.0)
        gray = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    if abs(GAMMA - 1.0) > 1e-6:
        inv = 1.0 / GAMMA
        table = (np.linspace(0,1,256) ** inv) * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        gray  = table[gray]
    # กลับเป็น 3 แชนเนลสำหรับโมเดลหลัก
    rgb = np.dstack([gray, gray, gray])
    return Image.fromarray(rgb, mode='RGB')

def list_images(folder):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files

# ---------- midline area decision ----------
def to_dark_mask(roi_gray):
    g = roi_gray.copy()
    # ใช้ OTSU + invert ให้เส้นมืดเป็น 1
    _, mask = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # เปิดรูเล็กๆ ลด noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(np.float32)  # 0/1

def dark_fraction_below(mask, mid_ratio=0.5, ignore_margin=0.10):
    H, W = mask.shape
    y_mid = int(round(mid_ratio*(H-1)))
    x0 = int(round(ignore_margin*W)); x1 = W - x0
    if x1 <= x0: x0, x1 = 0, W
    roi = mask[:, x0:x1]
    tot = roi.sum() + 1e-6
    below = roi[y_mid+1:, :].sum()
    frac = float(below / tot)  # 0..1
    return frac, y_mid, (x0, x1)

def decide_with_midline(pil_img_rgb, pair_model):
    # ใช้ภาพเดิม (หลัง enhance แล้ว) ทำเป็น gray -> resize สำหรับ pair
    g_full = np.array(pil_img_rgb.convert("L"))
    roi_gray = cv2.resize(g_full, (W, H), interpolation=cv2.INTER_LINEAR)

    # โมเดล pair: ให้ a (0..9) และ b=(a+1)%10
    x = (roi_gray.astype(np.float32)/255.0)[None, ..., None]  # [1,H,W,1]
    prob = pair_model.predict(x, verbose=0)[0]                 # [10]
    a = int(prob.argmax()); b=(a+1)%10; conf=float(prob[a])

    # เกณฑ์ตัดสินพื้นที่ใต้เส้น
    mask = to_dark_mask(roi_gray)
    frac, y_mid, (x0,x1) = dark_fraction_below(mask, MID_RATIO, IGNORE_X_MARGIN)
    thr_used = PAIR_THR.get(a, AREA_THR)
    d1_disc = b if (conf >= PAIR_CONF_MIN and frac >= thr_used) else a


    # (option) เซฟกราฟ
    if SAVE_PLOTS or SHOW_PLOT:
        Hh, Ww = roi_gray.shape
        fig = plt.figure(figsize=(9,5))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(pil_img_rgb); ax1.axis("off")
        ax1.set_title(f"{a}->{b} conf={conf:.2f}")
        ax1.axhline(y_mid, color='C0', lw=3)
        ax1.add_patch(plt.Rectangle((x0,0), x1-x0, Hh, fill=False, lw=2, ec="yellow"))

        ax2 = fig.add_subplot(1,2,2)
        row_dark = mask[:, x0:x1].sum(axis=1)
        y = np.arange(Hh)
        ax2.plot(row_dark, y, label="Row darkness")
        ax2.invert_yaxis()
        ax2.axhline(y_mid, color='C0', lw=2)
        ax2.set_xlabel("Darkness"); ax2.set_ylabel("Y")
        ax2.set_title(f"area_below={frac:.3f} | thr={thr_used:.2f} | d1_disc={d1_disc}")
        ax2.legend()
        fig.tight_layout()

        if SAVE_PLOTS:
            os.makedirs(PLOT_DIR, exist_ok=True)
            # ชื่อไฟล์จะถูกตั้งภายนอก (ในลูปหลัก)
            fig._decide_plot_figure = fig  # ติดธงไว้
        if SHOW_PLOT:
            plt.show()
        else:
            plt.close(fig)

    return d1_disc, a, b, conf, frac

# --------------- MAIN ----------------
def main():
    # โหลดโมเดล
    model_digit = tf.keras.models.load_model(MODEL_DIGIT)
    model_pair  = tf.keras.models.load_model(MODEL_PAIR, compile=False)

    files = list_images(TEST_IMAGE_DIR)
    print(f"พบรูปทดสอบ {len(files)} ภาพ")

    rows = [["file","digit_pred","is_NaN","pair","pair_conf","area_below","thr_used","final_digit"]]

    for fp in files:
        pil = Image.open(fp)
        pil_enh = enhance_digit(pil)

        # ----- predict หลัก (0..9 + NaN=10) -----
        arr = np.array(pil_enh, dtype="float32")     # หมายเหตุ: โมเดลหลักของคุณเทรนแบบไม่หาร 255
        x = arr[None, ...]                           # (1, H, W, 3)
        pred = model_digit.predict(x, verbose=0)
        probs = pred[0]
        cls   = int(np.argmax(probs))
        digit_conf = float(np.max(probs))
        is_nan = (cls == 10) if (probs.shape[0] == 11) else False

        # ถ้าไม่ใช่ NaN และมั่นใจพอ → ใช้ผลโมเดลหลักทันที
        if (not is_nan) and (digit_conf >= DIGIT_CONF_MIN):
            final_digit = cls
            print(f"{os.path.basename(fp)} -> Predict: {cls} (conf={digit_conf:.2f}) [main model]")
            rows.append([os.path.basename(fp), cls, 0, "", "", "", "", final_digit])
            continue

# มิฉะนั้น (NaN หรือความมั่นใจต่ำ) → ไป pair + midline


        # ----- ถ้า NaN → ใช้ pair + midline -----
        d1_disc, a, b, conf, frac = decide_with_midline(pil_enh, model_pair)
        thr_used = PAIR_THR.get(a, AREA_THR)
        final_digit = d1_disc

        print(f"{os.path.basename(fp)} -> Predict: NaN  | pair {a}->{b} conf={conf:.2f} | "
              f"area_below={frac:.3f} thr={thr_used:.2f} => final={final_digit}")

        # เซฟกราฟ (ถ้าตั้งไว้)
        if SAVE_PLOTS:
            # สร้างกราฟใหม่อีกครั้งแบบ headless เพื่อเซฟ (กรณี figure ถูกปิดไปแล้ว)
            g = np.array(pil_enh.convert("L"))
            roi_gray = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = to_dark_mask(roi_gray)
            frac2, y_mid, (x0,x1) = dark_fraction_below(mask, MID_RATIO, IGNORE_X_MARGIN)

            Hh, Ww = roi_gray.shape
            fig = plt.figure(figsize=(9,5))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(pil_enh); ax1.axis("off")
            ax1.set_title(f"{a}->{b} conf={conf:.2f}")
            ax1.axhline(y_mid, color='C0', lw=3)
            ax1.add_patch(plt.Rectangle((x0,0), x1-x0, Hh, fill=False, lw=2, ec="yellow"))
            ax2 = fig.add_subplot(1,2,2)
            row_dark = mask[:, x0:x1].sum(axis=1)
            y = np.arange(Hh)
            ax2.plot(row_dark, y, label="Row darkness")
            ax2.invert_yaxis()
            ax2.axhline(y_mid, color='C0', lw=2)
            ax2.set_xlabel("Darkness"); ax2.set_ylabel("Y")
            ax2.set_title(f"area_below={frac2:.3f} | thr={thr_used:.2f} | d1_disc={final_digit}")
            ax2.legend()
            fig.tight_layout()
            os.makedirs(PLOT_DIR, exist_ok=True)
            out_path = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(fp))[0] + "_midline.png")
            fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            print("  -> saved plot:", out_path)

        rows.append([os.path.basename(fp), "NaN", 1, f"{a}->{b}", f"{conf:.4f}",
                     f"{frac:.6f}", f"{PAIR_THR.get(a, AREA_THR):.2f}", final_digit])

    # ----- เซฟ CSV สรุป -----
    if SAVE_CSV:
        os.makedirs(os.path.dirname(SAVE_CSV) or ".", exist_ok=True)
        with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("Wrote CSV ->", SAVE_CSV, "rows:", len(rows)-1)

if __name__ == "__main__":
    main()
