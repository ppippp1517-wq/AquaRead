import os, re, glob, csv, argparse, numpy as np
from PIL import Image
import tensorflow as tf
import torch, torch.nn as nn

# ---------- Keras: เดาคู่เลข a->b (10 คลาส = a) ----------
def load_pair_model(h5_path: str):
    return tf.keras.models.load_model(h5_path, compile=False)

def pair_predict(model, pil: Image.Image, W=20, H=32):
    im = pil.convert("L").resize((W, H), Image.BILINEAR)
    x  = (np.array(im, dtype=np.float32) / 255.0)[None, ..., None]  # [1,H,W,1]
    p  = model.predict(x, verbose=0)[0]                              # [10]
    a  = int(p.argmax())                                             # 0..9 (= a)
    b  = (a + 1) % 10
    conf = float(p[a])
    return a, b, conf

# ---------- PyTorch: phase (ความคืบหน้า 0..1) ----------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x):                    # x: [B,1,H,W]
        f = self.net(x).view(x.size(0), -1)  # [B,64]
        return self.proj(f)                  # [B,128]

class CNNBiLSTMPhase(nn.Module):
    def __init__(self, feat=128, hid=128, layers=1):
        super().__init__()
        self.cnn  = SmallCNN(feat)
        self.lstm = nn.LSTM(feat, hid, num_layers=layers,
                            bidirectional=True, batch_first=False)
        self.head = nn.Linear(hid*2, 1)
    def forward(self, x):                    # x: [T,B,1,H,W]
        T,B = x.size(0), x.size(1)
        f = self.cnn(x.view(T*B, 1, x.size(3), x.size(4))).view(T,B,-1)
        y,_ = self.lstm(f)                   # [T,B,2*hid]
        return torch.sigmoid(self.head(y)).squeeze(-1)  # [T,B]

def load_phase_model(ckpt_path: str, device):
    m = CNNBiLSTMPhase().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model"])
    return m

def load_gray_tensor(pil: Image.Image, W: int, H: int, keep_ratio=True):
    im = pil.convert("L")
    if keep_ratio:
        w,h = im.size
        s = min(W/w, H/h)
        nw,nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
        im = im.resize((nw,nh), Image.BILINEAR)
        canvas = Image.new("L",(W,H), color=0)
        off = ((W-nw)//2, (H-nh)//2)
        canvas.paste(im, off)
        im = canvas
    else:
        im = im.resize((W,H), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

def soft_from_phi(phi: float, low=0.4, high=0.6):
    if phi <= low:  return 0.0
    if phi >= high: return 1.0
    return (phi - low) / (high - low)

def natural_key(p):
    base = os.path.basename(p)
    nums = re.findall(r'\d+', base)
    return [int(n) for n in nums] if nums else [base]

def collect_files(single_img, globs):
    files = []
    if globs:
        for pat in globs:
            files.extend(glob.glob(pat))
    if single_img:
        files.append(single_img)
    files = sorted(set(files), key=natural_key)
    files = [f for f in files if os.path.isfile(f)]
    return files

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Read pair(a->b) + phase (single or batch) + optional CSV")
    ap.add_argument("--h5",   required=True, help="โมเดล Keras pair_ab_keras.h5")
    ap.add_argument("--ckpt", required=True, help="checkpoint phase_sanity.pt")
    ap.add_argument("--img",  help="ไฟล์ภาพเดี่ยว")
    ap.add_argument("--glob", nargs="+", help="แพทเทิร์นหลายไฟล์ เช่น D:/.../output_*.png หรือใส่ชื่อไฟล์ตรงๆ ก็ได้")
    ap.add_argument("--csv",  help="พาธไฟล์ .csv สำหรับบันทึกผล (ออปชัน)")
    ap.add_argument("--img_w", type=int, default=20)
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--keep_ratio", action="store_true")
    ap.add_argument("--repeat", type=int, default=16)
    ap.add_argument("--low",  type=float, default=0.4)
    ap.add_argument("--high", type=float, default=0.6)
    args = ap.parse_args()

    files = collect_files(args.img, args.glob)
    if not files:
        raise SystemExit("ไม่พบไฟล์ภาพ: ใช้ --img <ไฟล์> หรือ --glob <แพทเทิร์น...>")

    # โหลดโมเดลครั้งเดียว
    pair_model = load_pair_model(args.h5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_model = load_phase_model(args.ckpt, device)

    rows = [["path","file","a","b","conf_pair","phi","s","d1_now","d1_disc"]]

    for fp in files:
        pil = Image.open(fp)
        a,b,conf = pair_predict(pair_model, pil, args.img_w, args.img_h)

        x1 = load_gray_tensor(pil, args.img_w, args.img_h, args.keep_ratio).unsqueeze(1)  # [1,1,H,W]
        x  = x1.repeat(args.repeat, 1, 1, 1, 1).to(device)
        with torch.no_grad():
            phi = float(phase_model(x).mean().item())
        s = soft_from_phi(phi, args.low, args.high)
        d1_now  = (a + s) % 10
        d1_disc = b if phi >= 0.5 else a

        print(os.path.basename(fp),
              f"| pair {a}->{b} (conf={conf:.3f})",
              f"| phi={phi:.3f} s={s:.3f}",
              f"| d1_now={d1_now:.3f}",
              f"| d1_disc={d1_disc}")

        rows.append([fp, os.path.basename(fp), a, b,
                     f"{conf:.6f}", f"{phi:.6f}", f"{s:.6f}", f"{d1_now:.6f}", d1_disc])

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("Wrote CSV ->", args.csv, "rows:", len(rows)-1)

if __name__ == "__main__":
    main()
