
import os, re, glob, argparse, random
from typing import List, Tuple
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# ---------------- Utilities ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_seq_dirs(root: str) -> List[Tuple[str,int]]:
    pat = os.path.join(root, "*_to_*", "*")
    seqs = [d for d in glob.glob(pat) if os.path.isdir(d)]
    seqs.sort()
    out = []
    for sd in seqs:
        m = re.search(r'([0-9])_to_([0-9])', sd.replace('\\','/'))
        if not m: 
            continue
        a, b = int(m.group(1)), int(m.group(2))
        # pair id = a  (0->1=0, ..., 9->0=9)
        out.append((sd, a))
    return out

def natural_key(p):
    base = os.path.basename(p)
    nums = re.findall(r'\d+', base)
    return [int(n) for n in nums] if nums else [base]

def resize_keep(im: Image.Image, W:int, H:int):
    w,h = im.size
    if w==0 or h==0:
        return Image.new("L",(W,H),color=0)
    s=min(W/w, H/h)
    nw,nh=max(1,int(round(w*s))), max(1,int(round(h*s)))
    im = im.resize((nw,nh), Image.BILINEAR)
    canvas = Image.new("L",(W,H), color=0)
    off = ((W-nw)//2, (H-nh)//2)
    canvas.paste(im, off)
    return canvas

# ---------------- Dataset ----------------
class PairFrameDataset(Dataset):
    def __init__(self, seq_dirs: List[Tuple[str,int]], img_w=20, img_h=32, keep_ratio=True, aug=True):
        self.items = []  # (img_path, pair_id=a)
        self.img_w, self.img_h = img_w, img_h
        self.keep_ratio, self.aug = keep_ratio, aug
        for sd, a in seq_dirs:
            files = [os.path.join(sd, fn) for fn in os.listdir(sd) if os.path.splitext(fn)[1].lower() in IMG_EXT]
            files.sort(key=natural_key)
            for p in files:
                self.items.append((p, a))

    def __len__(self): return len(self.items)

    def _augment(self, im: Image.Image):
        # light aug: brightness/contrast, vertical shift, tiny blur
        if random.random() < 0.7:
            a = 0.9 + 0.2*random.random()  # contrast
            b = (random.random()-0.5)*0.2  # brightness shift [-0.1,0.1]
            arr = np.array(im, dtype=np.float32)/255.0
            arr = np.clip(a*arr + b, 0, 1)
            im  = Image.fromarray((arr*255).astype(np.uint8))
        if random.random() < 0.5:
            dy = random.randint(-2,2)
            im = Image.fromarray(np.roll(np.array(im), dy, axis=0))
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=0.3))
        return im

    def __getitem__(self, idx):
        path, a = self.items[idx]
        im = Image.open(path).convert("L")
        if self.keep_ratio:
            im = resize_keep(im, self.img_w, self.img_h)
        else:
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
        if self.aug:
            im = self._augment(im)
        arr = np.array(im, dtype=np.float32)/255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x, a

# ---------------- Model ----------------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x):  # [B,1,H,W]
        f = self.net(x)
        f = f.view(f.size(0), -1)
        return self.proj(f)

class PairABNet(nn.Module):
    """
    Triple-view: full + top + bottom (resized back) -> concat -> classifier(10)
    """
    def __init__(self, feat=64, top_ratio=0.55, bot_ratio=0.55, D=64):
        super().__init__()
        self.top_ratio, self.bot_ratio = top_ratio, bot_ratio
        self.backbone = SmallCNN(out_dim=D)
        self.cls = nn.Sequential(nn.Linear(D*3, feat), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(feat, 10))

    def _crop_top_bot(self, x):
        # x: [B,1,H,W]
        H = x.size(2)
        th = int(round(H*self.top_ratio))
        bh = int(round(H*self.bot_ratio))
        top = x[:, :, :th, :]
        bot = x[:, :, H-bh:, :]
        # resize to original size for backbone
        top = F.interpolate(top, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        bot = F.interpolate(bot, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        return top, bot

    def forward(self, x):
        top, bot = self._crop_top_bot(x)
        f_full = self.backbone(x)
        f_top  = self.backbone(top)
        f_bot  = self.backbone(bot)
        f = torch.cat([f_full, f_top, f_bot], dim=-1)
        logits = self.cls(f)     # [B,10]  => pair id = a
        return logits

# ---------------- Train / Eval ----------------
@torch.no_grad()
def accuracy(logits, y):
    pred = logits.argmax(-1)
    return (pred == y).float().mean().item()

def split_by_sequence(root, val_split=0.2, seed=42):
    seqs = list_seq_dirs(root)
    random.Random(seed).shuffle(seqs)
    n_val = max(1, int(round(len(seqs)*val_split)))
    va = seqs[:n_val]; tr = seqs[n_val:]
    return tr, va

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tr_seqs, va_seqs = split_by_sequence(args.root, args.val_split)
    tr_ds = PairFrameDataset(tr_seqs, args.img_w, args.img_h, args.keep_ratio, aug=True)
    va_ds = PairFrameDataset(va_seqs, args.img_w, args.img_h, args.keep_ratio, aug=False)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    model = PairABNet(top_ratio=args.top_ratio, bot_ratio=args.bot_ratio).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for ep in range(1, args.epochs+1):
        # train
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        tr_loss = tot / max(1, n)

        # val
        model.eval()
        v_loss, v_acc, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                v_loss += loss.item()
                v_acc  += accuracy(logits, y)
                vn += 1
        v_loss /= max(1, vn); v_acc /= max(1, vn)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {v_loss:.4f} | val_acc {v_acc*100:.2f}%")

        if v_acc > best:
            best = v_acc
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
            print(f"  ✔ Save best ({best*100:.2f}%) to {args.save}")

    print("Done. Best val_acc = %.2f%%" % (best*100))

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = PairABNet(top_ratio=args.top_ratio, bot_ratio=args.bot_ratio).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # load image
    im = Image.open(args.img).convert("L")
    if args.keep_ratio: im = resize_keep(im, args.img_w, args.img_h)
    else: im = im.resize((args.img_w, args.img_h), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32)/255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        a = int(probs.argmax()); b = (a+1)%10; conf = float(probs[a])
    print(f"{os.path.basename(args.img)} | pair {a}->{b} | conf={conf:.3f} | probs={np.round(probs,3)}")

def export_ts(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = PairABNet(top_ratio=args.top_ratio, bot_ratio=args.bot_ratio).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # dummy input
    x = torch.randn(1,1,args.img_h,args.img_w, device=device)
    ts = torch.jit.trace(model, x)
    ts.save(args.out)
    print("Saved TorchScript to", args.out)

def build_parser():
    p = argparse.ArgumentParser("Pair a->b classifier (single script)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--root", required=True)
    pt.add_argument("--img_w", type=int, default=20)
    pt.add_argument("--img_h", type=int, default=32)
    pt.add_argument("--keep_ratio", action="store_true")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--batch", type=int, default=256)
    pt.add_argument("--lr", type=float, default=5e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--num_workers", type=int, default=0)
    pt.add_argument("--val_split", type=float, default=0.2)
    pt.add_argument("--top_ratio", type=float, default=0.55)
    pt.add_argument("--bot_ratio", type=float, default=0.55)
    pt.add_argument("--save", type=str, default="pair_ab.pt")
    pt.set_defaults(func=train)

    # infer
    pi = sub.add_parser("infer")
    pi.add_argument("--ckpt", required=True)
    pi.add_argument("--img", required=True)
    pi.add_argument("--img_w", type=int, default=20)
    pi.add_argument("--img_h", type=int, default=32)
    pi.add_argument("--keep_ratio", action="store_true")
    pi.add_argument("--top_ratio", type=float, default=0.55)
    pi.add_argument("--bot_ratio", type=float, default=0.55)
    pi.set_defaults(func=infer)

    # export
    pe = sub.add_parser("export-ts")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--out", type=str, default="pair_ab.ts")
    pe.add_argument("--img_w", type=int, default=20)
    pe.add_argument("--img_h", type=int, default=32)
    pe.add_argument("--top_ratio", type=float, default=0.55)
    pe.add_argument("--bot_ratio", type=float, default=0.55)
    pe.set_defaults(func=export_ts)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
