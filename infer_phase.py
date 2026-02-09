
import os, glob, argparse, csv
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# ---------- Model must match the training definition ----------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):  # x: [B,1,H,W]
        f = self.net(x)     # [B,64,1,1]
        f = f.view(f.size(0), -1)  # [B,64]
        f = self.proj(f)           # [B,out_dim]
        return f

class CNNBiLSTMPhase(nn.Module):
    def __init__(self, feat_dim=128, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.cnn = SmallCNN(out_dim=feat_dim)
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
                             bidirectional=True, batch_first=False)
        self.head = nn.Linear(lstm_hidden*2, 1)

    def forward(self, x):
        # x: [T,B,1,H,W]
        T, B = x.size(0), x.size(1)
        x_ = x.view(T*B, x.size(2), x.size(3), x.size(4))
        f = self.cnn(x_)                 # [T*B,D]
        f = f.view(T, B, -1)             # [T,B,D]
        y, _ = self.lstm(f)              # [T,B,2H]
        phase = torch.sigmoid(self.head(y)).squeeze(-1)  # [T,B]
        return phase

# ---------- Image utilities ----------
def pad_to_size(img: Image.Image, target_wh: Tuple[int,int], bg=0):
    W,H = target_wh
    canvas = Image.new("L", (W,H), color=bg)
    w,h = img.size
    off = ((W-w)//2, (H-h)//2)
    canvas.paste(img, off)
    return canvas

def resize_keep_ratio(img: Image.Image, target_wh: Tuple[int,int]):
    W,H = target_wh
    w,h = img.size
    if w==0 or h==0:
        return Image.new("L",(W,H), color=0)
    scale = min(W/w, H/h)
    nw,nh = max(1,int(round(w*scale))), max(1,int(round(h*scale)))
    img = img.resize((nw,nh), Image.BILINEAR)
    return pad_to_size(img, (W,H))

def load_gray_as_tensor(path: str, W: int, H: int, keep_ratio: bool):
    im = Image.open(path).convert("L")
    if keep_ratio:
        im = resize_keep_ratio(im, (W,H))
    else:
        im = im.resize((W,H), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0   # [H,W]
    x = torch.from_numpy(arr).unsqueeze(0)         # [1,H,W]
    return x

# ---------- Hysteresis crossing detection ----------
def detect_crossing(phases: np.ndarray, low=0.4, high=0.6, sustain=2):
    """
    Return first index t where phase crosses upward from <=low to >=high and stays >=high for `sustain` frames.
    If not found, return -1.
    """
    below_low_seen = False
    for t in range(len(phases)):
        if phases[t] <= low:
            below_low_seen = True
        if below_low_seen and phases[t] >= high:
            # check sustain
            end = min(len(phases), t+sustain)
            if np.all(phases[t:end] >= high):
                return t
    return -1

# ---------- Main ----------
def list_seq_dirs(root: str) -> List[str]:
    pattern = os.path.join(root, "*_to_*", "*")
    seqs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    seqs.sort()
    return seqs

def natural_key(p: str):
    import re
    base = os.path.basename(p)
    nums = re.findall(r'\d+', base)
    return [int(n) for n in nums] if nums else [base]

def main():
    ap = argparse.ArgumentParser(description="Inference: predict phase per frame and export CSVs")
    ap.add_argument("--ckpt", required=True, help="path to phase_sanity.pt")
    ap.add_argument("--root", required=True, help="dataset root (contains *_to_*)")
    ap.add_argument("--out", required=True, help="output dir for CSVs")
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=20)
    ap.add_argument("--keep_ratio", action="store_true")
    ap.add_argument("--seq", type=str, default=None, help="optional: path to a single seq dir to run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = CNNBiLSTMPhase(feat_dim=128, lstm_hidden=128, lstm_layers=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.seq:
        seq_dirs = [args.seq]
    else:
        seq_dirs = list_seq_dirs(args.root)

    print(f"Found {len(seq_dirs)} sequences")
    for sd in seq_dirs:
        # collect images
        files = [os.path.join(sd, fn) for fn in os.listdir(sd) if os.path.splitext(fn)[1].lower() in IMG_EXTS]
        files.sort(key=natural_key)
        if len(files) == 0:
            print(f"[skip] No images in {sd}")
            continue

        # build tensor [T,1,1,H,W]
        frames = [load_gray_as_tensor(p, args.img_w, args.img_h, args.keep_ratio) for p in files]
        x = torch.stack(frames, dim=0).unsqueeze(1).to(device)

        with torch.no_grad():
            phases = model(x).squeeze(1).detach().cpu().numpy()  # [T]

        # write CSV next to out dir preserving subfolders
        rel = os.path.relpath(sd, args.root)
        out_dir = os.path.join(args.out, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(args.out, rel + ".csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx","filename","phase"])
            for i,(p,ph) in enumerate(zip(files, phases)):
                w.writerow([i, os.path.basename(p), float(ph)])

        # detect crossing
        t_cross = detect_crossing(phases, low=0.4, high=0.6, sustain=2)
        if t_cross >= 0:
            print(f"[{rel}] crossing at frame {t_cross} -> {os.path.basename(files[t_cross])}, phase={phases[t_cross]:.3f}")
        else:
            print(f"[{rel}] no stable crossing detected")

    print("Done.")

if __name__ == "__main__":
    main()
