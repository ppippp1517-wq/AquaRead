
# PCIC CNN+BiLSTM v2 - READY VERSION
# ------------------------------------------------------------
# - Dataset: <x>to<y> folders under --root (e.g., 0to1, 3 to 4, 9to0)
# - Labels: digit=0..9; transition frames (phase in (EDGE_LO, EDGE_HI))
#           are MASKED OUT from digit CE, phase head learns all frames.
# - Split: sequence-level (by folder), avoids leakage.
# - Train: AdamW + OneCycleLR, light augmentation.
# - Save: last + best checkpoints.
# - Stream: phase-or-stability gate + ANCHOR-BASE (no cumulative add).
# - CLI: most knobs can be changed via arguments.
# ------------------------------------------------------------
import os, re, math, time, random, argparse
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------- Utils ---------------------------
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# ------------------------- Dataset ---------------------------
_digit_folder_re = re.compile(r'(\d)\s*to\s*(\d)', re.IGNORECASE)

def load_image_letterbox(path: str, size_hw: Tuple[int,int], keep_ratio: bool=True) -> Image.Image:
    im = Image.open(path).convert('L')
    th, tw = size_hw
    if keep_ratio:
        im = ImageOps.contain(im, (tw, th), Image.BILINEAR)
        canvas = Image.new('L', (tw, th), 128)
        x = (tw - im.width)//2
        y = (th - im.height)//2
        canvas.paste(im, (x, y))
        return canvas
    else:
        return im.resize((tw, th), Image.BILINEAR)

def to_tensor_norm(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[None, ...]
    return torch.from_numpy(arr)

def augment_light(im: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(0.9 + 0.2*random.random())
    if random.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(0.9 + 0.2*random.random())
    if random.random() < 0.3:
        im = im.rotate(random.uniform(-4,4), resample=Image.BILINEAR, expand=False, fillcolor=128)
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0,0.6)))
    return im

class PCICSequence:
    def __init__(self, name: str, x: int, y: int, frames: List[str]):
        self.name = name
        self.x = x
        self.y = y
        self.frames = frames
    def __len__(self): return len(self.frames)
    def phase(self, idx: int) -> float:
        n = len(self.frames)
        return 0.0 if n<=1 else idx/(n-1)

def discover_sequences(root: str) -> List[PCICSequence]:
    seqs: List[PCICSequence] = []
    for name in sorted(os.listdir(root), key=natural_key):
        m = _digit_folder_re.search(name)
        if not m: continue
        x = int(m.group(1)); y = int(m.group(2))
        folder = os.path.join(root, name)
        if not os.path.isdir(folder): continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        if not files: continue
        files.sort(key=natural_key)
        seqs.append(PCICSequence(name, x, y, files))
    return seqs

class PCICDataset(Dataset):
    """Returns: clip[T,1,H,W], digit_labels[T], digit_mask[T], phases[T,1]"""
    def __init__(self, sequences: List[PCICSequence], include_names: List[str],
                 img_size=(20,32), window_len=16, stride=4,
                 edge_lo=0.10, edge_hi=0.90, train=True, keep_ratio=True):
        self.seqs = [s for s in sequences if s.name in set(include_names)]
        self.H, self.W = img_size
        self.window = window_len
        self.stride = stride
        self.edge_lo = edge_lo
        self.edge_hi = edge_hi
        self.train = train
        self.keep_ratio = keep_ratio

        self.index = []
        for sid, s in enumerate(self.seqs):
            n = len(s)
            if n < self.window: continue
            for st in range(0, n - self.window + 1, self.stride):
                self.index.append((sid, st))
        if not self.index:
            raise RuntimeError("No windows found. Check data and parameters.")

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        sid, st = self.index[i]
        s = self.seqs[sid]
        clips=[]; digit=[]; mask=[]; phases=[]
        for t in range(self.window):
            idx = st+t
            im = load_image_letterbox(s.frames[idx], (self.H,self.W), keep_ratio=self.keep_ratio)
            if self.train: im = augment_light(im)
            x = to_tensor_norm(im)
            clips.append(x)

            p = s.phase(idx)
            phases.append([p])
            if p < self.edge_lo:
                d = s.x; m=1
            elif p > self.edge_hi:
                d = s.y; m=1
            else:
                d = s.x; m=0
            digit.append(d); mask.append(m)

        clip = torch.stack(clips, dim=0)
        digit = torch.tensor(digit, dtype=torch.long)
        mask  = torch.tensor(mask, dtype=torch.float32)
        phases= torch.tensor(np.array(phases, dtype=np.float32))
        return clip, digit, mask, phases

# ------------------------- Model ----------------------------
class TinyEncoder(nn.Module):
    def __init__(self, in_ch=1, feat=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, 2, 1),    nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, feat, 3, 2, 1),  nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):  # (B,T,1,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.backbone(x)
        f = self.gap(f).view(B*T, -1)
        return f.view(B, T, -1)

class CNNLSTM(nn.Module):
    def __init__(self, in_ch=1, feat=128, hidden=128, num_layers=2, num_classes=10):
        super().__init__()
        self.encoder = TinyEncoder(in_ch=in_ch, feat=feat)
        self.lstm = nn.LSTM(input_size=feat, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.head_digit = nn.Linear(hidden*2, num_classes)
        self.head_phase = nn.Linear(hidden*2, 1)
    def forward(self, x):
        f = self.encoder(x)
        y, _ = self.lstm(f)
        logits = self.head_digit(y)
        phase  = torch.sigmoid(self.head_phase(y))
        return logits, phase

# ------------------------- Loss/Eval -------------------------
def compute_loss(logits, phase_pred, labels, mask, phase_gt, lambda_phase:float):
    ce_all = F.cross_entropy(logits.view(-1,10), labels.view(-1), reduction='none')
    ce_masked = (ce_all * mask.view(-1)).sum() / (mask.sum() + 1e-6)
    l1 = F.smooth_l1_loss(phase_pred, phase_gt)
    return ce_masked + lambda_phase*l1, (ce_masked.item(), l1.item())

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for clip, labels, mask, phases in loader:
            clip, labels, mask = clip.to(device), labels.to(device), mask.to(device)
            logits, _ = model(clip)
            pred = logits.argmax(-1)
            m = mask.bool()
            correct += (pred[m]==labels[m]).sum().item()
            total   += m.sum().item()
    return correct/max(total,1)

# ------------------------- Stream ---------------------------
class StreamReader:
    def __init__(self, model, device, window_len=16, stride=2,
                 phase_thresh=0.80, decimal_shift=0,
                 stable_k=3, conf_thresh=0.80, anchor_base: float=0.0):
        self.model = model.eval(); self.device=device
        self.buf = deque(maxlen=window_len); self.stride=stride
        self.phase_thresh=phase_thresh
        self.decimal_shift=decimal_shift; self.unit = 10**(decimal_shift)
        self.stable_need=stable_k; self.conf_thresh=conf_thresh
        self.anchor_base=anchor_base
        self.prev_value=None
        self.last_digit=None; self.stable_k=0
        self._last_print=None

    @torch.no_grad()
    def push(self, arr_1xHxW: np.ndarray):
        self.buf.append(torch.from_numpy(arr_1xHxW).unsqueeze(0))
        if len(self.buf)<self.buf.maxlen or (len(self.buf)%self.stride)!=0:
            return None
        clip = torch.stack(list(self.buf), dim=1).to(self.device)
        logits, phase = self.model(clip)
        tail=4
        probs = logits[:, -tail:, :].softmax(-1).squeeze(0)  # (tail,10)
        phase_val = float(phase[:, -tail:, :].mean().item())
        confs, ids = probs.max(-1)
        pred_digit = int(ids.mode()[0].item())
        conf_avg = float(confs.mean().item())

        if self.last_digit is None or pred_digit!=self.last_digit:
            self.last_digit = pred_digit; self.stable_k = 1
        else:
            self.stable_k += 1

        accept = (phase_val >= self.phase_thresh) or (self.stable_k >= self.stable_need and conf_avg >= self.conf_thresh)
        value = self.anchor_base + pred_digit*self.unit if accept else (self.prev_value if self.prev_value is not None else self.anchor_base)

        if self._last_print is None or value != self._last_print:
            print(f"[stream] phase={phase_val:.2f} conf={conf_avg:.2f} pred={pred_digit} "
                  f"stable={self.stable_k} accept={accept} -> value={value}")
            self._last_print = value
        self.prev_value = value
        return value

# --------------------- Train / Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data & model
    ap.add_argument('--root', default=r'D:\projectCPE\pcic')
    ap.add_argument('--img_h', type=int, default=20)
    ap.add_argument('--img_w', type=int, default=32)
    ap.add_argument('--window_len', type=int, default=16)
    ap.add_argument('--window_stride', type=int, default=4)
    ap.add_argument('--edge_lo', type=float, default=0.10)
    ap.add_argument('--edge_hi', type=float, default=0.90)
    ap.add_argument('--keep_ratio', action='store_true', help='Use letterbox to keep aspect ratio (recommended).')

    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr_max', type=float, default=3e-3)
    ap.add_argument('--lambda_phase', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--save_dir', default='./checkpoints')

    # Stream/demo
    ap.add_argument('--demo', action='store_true', help='Run streaming demo after training.')
    ap.add_argument('--demo_seq', default='', help='Folder name to stream (e.g., 0to1). Default: first val seq.')
    ap.add_argument('--phase_thresh', type=float, default=0.80)
    ap.add_argument('--stable_k', type=int, default=3)
    ap.add_argument('--conf_thresh', type=float, default=0.80)
    ap.add_argument('--decimal_shift', type=int, default=0)
    ap.add_argument('--anchor_base', type=float, default=0.0)
    ap.add_argument('--stream_stride', type=int, default=2)

    args = ap.parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Discover sequences and split by folder
    seqs = discover_sequences(args.root)
    if not seqs:
        raise SystemExit(f'No sequences under {args.root}')
    idx = list(range(len(seqs)))
    random.shuffle(idx)
    n_val = max(1, int(len(seqs)*args.val_ratio))
    val_idx = set(idx[:n_val])
    val_names = [seqs[i].name for i in range(len(seqs)) if i in val_idx]
    train_names = [seqs[i].name for i in range(len(seqs)) if i not in val_idx]
    print(f'[SPLIT] train={len(train_names)} seq, val={len(val_names)} seq')
    print('[VAL] ', val_names)

    # Datasets
    train_ds = PCICDataset(seqs, train_names, img_size=(args.img_h,args.img_w),
                           window_len=args.window_len, stride=args.window_stride,
                           edge_lo=args.edge_lo, edge_hi=args.edge_hi, train=True,
                           keep_ratio=args.keep_ratio)
    val_ds   = PCICDataset(seqs, val_names, img_size=(args.img_h,args.img_w),
                           window_len=args.window_len, stride=args.window_stride,
                           edge_lo=args.edge_lo, edge_hi=args.edge_hi, train=False,
                           keep_ratio=args.keep_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # Model & optim
    model = CNNLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=1e-4)
    total_steps = max(1, len(train_loader)*args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr_max, total_steps=total_steps)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_last = os.path.join(args.save_dir, 'pcic_cnnlstm_v2.pt')
    ckpt_best = os.path.join(args.save_dir, 'pcic_cnnlstm_v2_best.pt')
    best_acc = -1.0

    # Train
    for ep in range(1, args.epochs+1):
        model.train()
        total=0.0; n=0
        for clip, labels, mask, phases in train_loader:
            clip = clip.to(device); labels=labels.to(device); mask=mask.to(device); phases=phases.to(device)
            opt.zero_grad()
            logits, phase_pred = model(clip)
            loss, (ce_m, l1) = compute_loss(logits, phase_pred, labels, mask, phases, args.lambda_phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item(); n+=1
        train_loss = total/max(n,1)
        val_acc = evaluate(model, val_loader, device)
        print(f'epoch {ep:02d}  loss={train_loss:.4f}  val_digit_acc={val_acc:.3f}')
        torch.save(model.state_dict(), ckpt_last)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_best)
            print(f'[BEST] acc={best_acc:.3f} -> {ckpt_best}')

    print(f'[OK] saved last={ckpt_last} best={ckpt_best}')

    # Demo stream
    if args.demo:
        # choose sequence
        val_seqs = [s for s in seqs if s.name in set(val_names)]
        if not val_seqs: val_seqs = seqs
        if args.demo_seq:
            matches = [s for s in val_seqs if s.name.lower()==args.demo_seq.lower()]
            if not matches:
                print(f"[WARN] demo_seq '{args.demo_seq}' not found. Using first val seq.")
                demo = val_seqs[0]
            else:
                demo = matches[0]
        else:
            demo = val_seqs[0]
        print(f'[DEMO] Streaming over: {demo.name}  frames={len(demo.frames)}')

        # load best
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        sr = StreamReader(model, device, window_len=args.window_len, stride=args.stream_stride,
                          phase_thresh=args.phase_thresh, decimal_shift=args.decimal_shift,
                          stable_k=args.stable_k, conf_thresh=args.conf_thresh,
                          anchor_base=args.anchor_base)
        def arr_from_path(p):
            im = load_image_letterbox(p, (args.img_h,args.img_w), keep_ratio=args.keep_ratio)
            return to_tensor_norm(im).numpy()
        for p in demo.frames:
            sr.push(arr_from_path(p))

if __name__ == '__main__':
    main()
