import os, glob, re
import numpy as np
import cv2, matplotlib.pyplot as plt
import tensorflow as tf

# ======= PATHS / CONFIG =======
MODEL_PAIR = r"D:/projectCPE/pair_ab_keras.h5"      # โมเดลคู่เลข (a->b)
IMG_INPUT  = r"D:/projectCPE/dataset/images/test"   # ใส่ได้ทั้ง "ไฟล์" หรือ "โฟลเดอร์"
SAVE_CSV   = r"D:/projectCPE/pair_threshold_results.csv"  # ถ้าไม่อยากเซฟ CSV -> ตั้งเป็น None

W, H   = 20, 32         # ขนาดอินพุต (width, height)
THR    = 0.60           # threshold ตัดสินใจเปลี่ยนเลขจาก a -> b
USE_HYST = False        # True = ใช้ฮิสเทอรีซิส; False = ใช้ phi ดิบ
LOW, HIGH = 0.40, 0.60  # ฮิสเทอรีซิส (ใช้เมื่อ USE_HYST=True)

SHOW_PLOT      = True   # เปิดกราฟตัวอย่าง
SHOW_FIRST_N   = 1      # แสดงกราฟกี่รูปแรก (โฟลเดอร์ใหญ่อย่าตั้งเยอะ)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# ======= SAVE PLOTS =======
SAVE_PLOTS = True
PLOT_DIR   = r"D:/projectCPE/plots_pair_threshold"
PLOT_FMT   = "png"
PLOT_DPI   = 150

def natural_key(p):
    b = os.path.basename(p)
    nums = re.findall(r'\d+', b)
    return [int(n) for n in nums] if nums else [b.lower()]

def collect_paths(inp):
    if os.path.isdir(inp):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(inp, f"*{ext}"))
        files = sorted(set(files), key=natural_key)
        return files
    elif os.path.isfile(inp):
        return [inp]
    else:
        raise FileNotFoundError(f"ไม่พบไฟล์/โฟลเดอร์: {inp}")

def soft_from_phi(phi, low, high):
    if phi <= low:  return 0.0
    if phi >= high: return 1.0
    return (phi - low) / (high - low)

def process_one(path, pair_model):
    # ----- โหลดภาพ -----
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"อ่านรูปไม่ได้: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ----- เตรียมอินพุตโมเดลคู่เลข (1ch, HxW) -----
    roi = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)
    x   = roi.astype(np.float32)/255.0
    x   = x[None, ..., None]  # [1,H,W,1]

    # ----- ทำนายคู่เลข a->b -----
    prob = pair_model.predict(x, verbose=0)[0]  # [10]
    a    = int(np.argmax(prob))
    b    = (a + 1) % 10
    conf = float(prob[a])

    # ----- คำนวณ phase แบบง่ายจาก pixel sum แนวตั้ง -----
    pixel_sum = roi.sum(axis=1).astype(np.float32)  # ความยาว H
    max_val   = max(1.0, float(pixel_sum.max()))
    phi_line  = pixel_sum / max_val
    phi       = float(phi_line.mean())              # 0..1

    s = soft_from_phi(phi, LOW, HIGH) if USE_HYST else phi
    d1_now_raw  = (a + phi) % 10
    d1_now_soft = (a + s)   % 10
    d1_disc     = b if phi >= THR else a  # ตัดสินด้วย threshold เดียว

        # ----- แสดงผล/เซฟกราฟ -----
    need_show = SHOW_PLOT and process_one.count < SHOW_FIRST_N
    need_save = SAVE_PLOTS

    if need_show or need_save:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9,5))

        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img_rgb)
        ax1.set_title(f"{os.path.basename(path)}\n{a}->{b} conf={conf:.2f}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1,2,2)
        y = np.arange(H)
        ax2.plot(pixel_sum, y, label="Pixel Sum (Y)")
        ax2.axvline(THR*max_val, color="red", linestyle="--", label=f"THR*max ({THR:.2f})")
        ttl = f"phi={phi:.3f}"
        ttl += f"  s={s:.3f}" if USE_HYST else ""
        ttl += f"  |  d1_disc={d1_disc}"
        ax2.set_title(ttl)
        ax2.invert_yaxis()
        ax2.set_xlabel("Sum"); ax2.set_ylabel("Y"); ax2.legend()
        fig.tight_layout()

        if need_save:
            os.makedirs(PLOT_DIR, exist_ok=True)
            out_path = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(path))[0] + f"_plot.{PLOT_FMT}")
            fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
            # พิมพ์แจ้งไฟล์ที่เซฟ
            print("  -> saved plot:", out_path)

        if need_show:
            plt.show()
            process_one.count += 1
        else:
            plt.close(fig)


    # ----- พิมพ์ผลในเทอร์มินัล -----
    print(os.path.basename(path),
          f"| pair {a}->{b} (conf={conf:.3f})",
          f"| phi={phi:.3f}",
          f"| s={s:.3f}",
          f"| d1_now_raw={d1_now_raw:.3f}",
          f"| d1_now_soft={d1_now_soft:.3f}",
          f"| thr={THR:.2f} -> d1_disc={d1_disc}")

    return {
        "path": path, "file": os.path.basename(path),
        "a": a, "b": b, "conf_pair": conf,
        "phi": phi, "s": s,
        "d1_now_raw": d1_now_raw, "d1_now_soft": d1_now_soft,
        "d1_disc": d1_disc
    }
process_one.count = 0

def main():
    files = collect_paths(IMG_INPUT)
    if not files:
        print("ไม่พบรูปภาพในโฟลเดอร์ที่ระบุ"); return

    pair_model = tf.keras.models.load_model(MODEL_PAIR, compile=False)

    rows = []
    for fp in files:
        try:
            res = process_one(fp, pair_model)
            rows.append(res)
        except Exception as e:
            print(f"[WARN] ข้ามไฟล์ {fp} เนื่องจาก: {e}")

    if SAVE_CSV:
        import csv
        os.makedirs(os.path.dirname(SAVE_CSV) or ".", exist_ok=True)
        with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path","file","a","b","conf_pair","phi","s","d1_now_raw","d1_now_soft","d1_disc"])
            for r in rows:
                w.writerow([r["path"], r["file"], r["a"], r["b"],
                            f"{r['conf_pair']:.6f}", f"{r['phi']:.6f}", f"{r['s']:.6f}",
                            f"{r['d1_now_raw']:.6f}", f"{r['d1_now_soft']:.6f}", r["d1_disc"]])
        print("Wrote CSV ->", SAVE_CSV, "rows:", len(rows))

if __name__ == "__main__":
    main()
