import os
import re
import glob
import math
import argparse
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ------------------------------
# Dataset: อ่านโฟลเดอร์ .../<a>_to_<b>/seq_xxxx/*.jpg
# ให้ label phase แบบอ่อนๆ: y_t = t/(T-1) ภายในแต่ละ sequence
# ------------------------------
class TransitionSeqDataset(Dataset):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, img_h: int = 32, img_w: int = 32, keep_ratio: bool = False):
        self.root = root
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio = keep_ratio

        # หา path ทุก seq: */*_to_*/*
        pattern = os.path.join(root, "*_to_*", "*")
        seq_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        seq_dirs.sort()

        self.samples = []  # list of (list_of_img_paths)
        for sd in seq_dirs:
            imgs = []
            for fn in sorted(os.listdir(sd)):
                ext = os.path.splitext(fn)[1].lower()
                if ext in self.IMG_EXT:
                    imgs.append(os.path.join(sd, fn))
            if len(imgs) >= 2:
                self.samples.append(imgs)

        if not self.samples:
            raise FileNotFoundError(f"ไม่พบลำดับภาพภายใต้: {pattern}.\\nโปรดเช็คโครงสร้างโฟลเดอร์เช่น D:/.../0_to_1/seq_0001/000.jpg ...")

    def __len__(self):
        return len(self.samples)

    def _load_and_resize(self, path: str):
        im = Image.open(path).convert("L")  # grayscale
        if self.keep_ratio:
            im = self._resize_keep_ratio(im, (self.img_w, self.img_h))
        else:
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
    # ใช้ numpy -> torch (แก้ TypeError)
        arr = np.array(im, dtype=np.float32) / 255.0  # [H,W] 0..1
        x = torch.from_numpy(arr).unsqueeze(0)        # [1,H,W]
        return x

    @staticmethod
    def _pad_to_size(img: Image.Image, target_wh: Tuple[int, int], bg=0):
        W, H = target_wh
        canvas = Image.new("L", (W, H), color=bg)
        w, h = img.size
        offset = ((W - w) // 2, (H - h) // 2)
        canvas.paste(img, offset)
        return canvas

    def _resize_keep_ratio(self, img: Image.Image, target_wh: Tuple[int, int]):
        W, H = target_wh
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("L", (W, H), color=0)
        scale = min(W / w, H / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        img = self._pad_to_size(img, (W, H))
        return img

    def __getitem__(self, idx):
        paths = self.samples[idx]
        frames = [self._load_and_resize(p) for p in paths]
        x = torch.stack(frames, dim=0)  # [T,1,H,W]
        T = x.size(0)
        if T == 1:
            y = torch.tensor([0.0], dtype=torch.float32)
        else:
            y = torch.linspace(0.0, 1.0, steps=T, dtype=torch.float32)  # y_t = t/(T-1)
        mask = torch.ones(T, dtype=torch.float32)
        return x, y, mask, paths[0]


def pad_collate(batch):
    # batch: list of (x[T,1,H,W], y[T], mask[T], tag)
    T_max = max(item[0].size(0) for item in batch)
    B = len(batch)
    C, H, W = batch[0][0].size(1), batch[0][0].size(2), batch[0][0].size(3)

    x_pad = torch.zeros(T_max, B, C, H, W, dtype=torch.float32)
    y_pad = torch.zeros(T_max, B, dtype=torch.float32)
    m_pad = torch.zeros(T_max, B, dtype=torch.float32)
    tags = []

    for b, (x, y, m, tag) in enumerate(batch):
        T = x.size(0)
        x_pad[:T, b] = x
        y_pad[:T, b] = y
        m_pad[:T, b] = m
        tags.append(tag)

    return x_pad, y_pad, m_pad, tags

# ------------------------------
# โมเดล: CNN encoder (เล็กๆ) + BiLSTM + หัว phase (sigmoid)
# ------------------------------
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

# ------------------------------
# Losses: Huber + temporal smoothness + monotonicity (ภายใน valid mask)
# ------------------------------
class PhaseLoss(nn.Module):
    def __init__(self, beta=0.1, lam_tv=0.1, lam_mono=0.2):
        super().__init__()
        self.beta = beta
        self.lam_tv = lam_tv
        self.lam_mono = lam_mono
        self.huber = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
        # pred,y,m: [T,B]
        # 1) regression on valid
        Lp = self.huber(pred, y)
        Lp = (Lp * m).sum() / m.sum().clamp_min(1.0)

        # 2) temporal smoothness & monotonicity on pairs (t,t+1) ที่ valid ทั้งคู่
        valid_pair = (m[1:] * m[:-1])
        if valid_pair.sum() > 0:
            dphi = pred[1:] - pred[:-1]
            Ltv = (dphi.abs() * valid_pair).sum() / valid_pair.sum()
            Lmono = (F.relu(-dphi) * valid_pair).sum() / valid_pair.sum()
        else:
            Ltv = pred.new_tensor(0.0)
            Lmono = pred.new_tensor(0.0)

        L = Lp + self.lam_tv * Ltv + self.lam_mono * Lmono
        return L, {"L": float(L.detach()), "L_phase": float(Lp.detach()), "L_tv": float(Ltv.detach()), "L_mono": float(Lmono.detach())}

# ------------------------------
# Train / Eval
# ------------------------------

def train_one_epoch(model, loss_fn, loader, opt, device):
    model.train()
    avg = {"L": 0.0, "L_phase": 0.0, "L_tv": 0.0, "L_mono": 0.0}
    n = 0
    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        L, stats = loss_fn(pred, y, m)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k in avg:
            avg[k] += stats.get(k, 0.0)
        n += 1
    for k in avg:
        avg[k] /= max(1, n)
    return avg

@torch.no_grad()
def eval_one_epoch(model, loss_fn, loader, device):
    model.eval()
    avg = {"L": 0.0, "L_phase": 0.0, "L_tv": 0.0, "L_mono": 0.0}
    n = 0
    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x)
        L, stats = loss_fn(pred, y, m)
        for k in avg:
            avg[k] += stats.get(k, 0.0)
        n += 1
    for k in avg:
        avg[k] /= max(1, n)
    return avg

# ------------------------------
# Main
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="CNN+BiLSTM Phase-only Sanity Trainer")
    ap.add_argument('--root', type=str, required=True, help='path ถึง dataset ที่มี *_to_*')
    ap.add_argument('--img_h', type=int, default=32)
    ap.add_argument('--img_w', type=int, default=32)
    ap.add_argument('--keep_ratio', action='store_true', help='resize แบบคงสัดส่วน + pad')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=0, help='Windows แนะนำ 0 ก่อน ถ้า Linux ใช้ 2-4 ได้')
    ap.add_argument('--save', type=str, default='phase_sanity.pt')
    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full = TransitionSeqDataset(args.root, img_h=args.img_h, img_w=args.img_w, keep_ratio=args.keep_ratio)
    N = len(full)
    n_val = max(1, int(round(N * args.val_split)))
    n_train = N - n_val
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type=='cuda'), collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type=='cuda'), collate_fn=pad_collate)

    model = CNNBiLSTMPhase(feat_dim=128, lstm_hidden=128, lstm_layers=1).to(device)
    loss_fn = PhaseLoss(beta=0.1, lam_tv=0.1, lam_mono=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, loss_fn, train_loader, opt, device)
        va = eval_one_epoch(model, loss_fn, val_loader, device)
        print(f"Epoch {ep:03d} | train L={tr['L']:.4f} (phase {tr['L_phase']:.4f}, tv {tr['L_tv']:.4f}, mono {tr['L_mono']:.4f}) | "
              f"val L={va['L']:.4f}")

        if va['L'] < best_val:
            best_val = va['L']
            torch.save({'model': model.state_dict(), 'args': vars(args)}, args.save)
            print(f"  ✔ Save best to {args.save}")

    print("เสร็จสิ้น. ใช้ไฟล์ checkpoint เพื่อ inference ต่อได้ครับ")

if __name__ == '__main__':
    main()
