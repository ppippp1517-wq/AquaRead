
import os, glob, argparse, re
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============== Dataset ==============
class DigitPhaseDataset(Dataset):
    IMG_EXT = {".jpg",".jpeg",".png",".bmp"}

    def __init__(self, root: str, img_h=32, img_w=20, keep_ratio=True, low=0.4, high=0.6):
        self.root = root
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio = keep_ratio
        self.low, self.high = low, high

        # discover sequences
        pat = os.path.join(root, "*_to_*", "*")
        seq_dirs = [d for d in glob.glob(pat) if os.path.isdir(d)]
        seq_dirs.sort()
        self.samples = []  # list of (paths, a, b, pair_id)
        for sd in seq_dirs:
            a,b = self._parse_pair(sd)
            if a is None: 
                continue
            imgs = [os.path.join(sd, fn) for fn in sorted(os.listdir(sd)) if os.path.splitext(fn)[1].lower() in self.IMG_EXT]
            if len(imgs) >= 2:
                self.samples.append((imgs, a, b, a))  # pair_id = a (0->1 -> 0, 1->2 -> 1, ..., 9->0 -> 9)
        if not self.samples:
            raise FileNotFoundError(f"ไม่พบข้อมูลในรูปแบบ {root}/*_to_*/*/*.png")

    @staticmethod
    def _parse_pair(path: str):
        m = re.search(r'([0-9])_to_([0-9])', path.replace('\\','/'))
        if not m: 
            return None, None
        a = int(m.group(1)); b = int(m.group(2))
        return a,b

    def __len__(self): return len(self.samples)

    def _resize_keep(self, im: Image.Image, W:int, H:int):
        w,h = im.size
        if w==0 or h==0:
            return Image.new("L",(W,H),color=0)
        s=min(W/w,H/h)
        nw,nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
        im = im.resize((nw,nh), Image.BILINEAR)
        canvas = Image.new("L",(W,H), color=0)
        off=((W-nw)//2,(H-nh)//2)
        canvas.paste(im,off)
        return canvas

    def _load(self, p):
        im = Image.open(p).convert("L")
        if self.keep_ratio:
            im = self._resize_keep(im, self.img_w, self.img_h)
        else:
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32)/255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x

    def __getitem__(self, idx):
        paths,a,b,pair_id = self.samples[idx]
        frames=[self._load(p) for p in paths]
        x = torch.stack(frames,0)  # [T,1,H,W]
        T=x.size(0)

        # phase label
        if T==1:
            y_phase = torch.tensor([0.0], dtype=torch.float32)
        else:
            y_phase = torch.linspace(0.0,1.0,steps=T, dtype=torch.float32)

        # digit class per frame: 0..9 + 10(transition)
        y_cls = torch.full((T,), 10, dtype=torch.long)  # default transition
        y_cls[y_phase <= self.low] = a
        y_cls[y_phase >= self.high] = b

        mask = torch.ones(T, dtype=torch.float32)
        tag  = paths[0]
        return x, y_phase, y_cls, mask, pair_id, tag

def pad_collate(batch):
    # batch of (x[T,1,H,W], y_phase[T], y_cls[T], mask[T], pair_id, tag)
    T_max = max(item[0].size(0) for item in batch)
    B = len(batch)
    C,H,W = batch[0][0].size(1), batch[0][0].size(2), batch[0][0].size(3)

    x_pad = torch.zeros(T_max,B,C,H,W)
    y_phase = torch.zeros(T_max,B)
    y_mask  = torch.zeros(T_max,B)
    y_cls   = torch.full((T_max,B), -100, dtype=torch.long)  # ignore_index padding
    pair_ids= torch.zeros(B, dtype=torch.long)
    tags=[]

    for b,(x,yp,yc,m,pid,tag) in enumerate(batch):
        T=x.size(0)
        x_pad[:T,b]=x; y_phase[:T,b]=yp; y_mask[:T,b]=m; y_cls[:T,b]=yc
        pair_ids[b]=pid; tags.append(tag)

    return x_pad, y_phase, y_cls, y_mask, pair_ids, tags

# ============== Model ==============
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(64,out_dim)
    def forward(self,x):
        f=self.net(x); f=f.view(f.size(0),-1); return self.proj(f)

class CNNBiLSTM_DigitPhase(nn.Module):
    def __init__(self, feat_dim=128, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.cnn = SmallCNN(out_dim=feat_dim)
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, num_layers=lstm_layers, bidirectional=True, batch_first=False)
        self.phase_head = nn.Linear(lstm_hidden*2, 1)
        self.digit_head = nn.Linear(lstm_hidden*2, 11)   # 0..9 + transition(10)
        self.pair_head  = nn.Linear(lstm_hidden*2, 10)   # 0->1 ... 9->0 (id = a)

    def forward(self, x, mask=None):
        # x: [T,B,1,H,W]
        T,B = x.size(0), x.size(1)
        x_ = x.view(T*B, x.size(2), x.size(3), x.size(4))
        f = self.cnn(x_)                  # [T*B,D]
        f = f.view(T,B,-1)                # [T,B,D]
        y,_ = self.lstm(f)                # [T,B,2H]

        phase = torch.sigmoid(self.phase_head(y)).squeeze(-1)   # [T,B]
        digit_logits = self.digit_head(y)                       # [T,B,11]

        if mask is None:
            h = y.mean(0)                                       # [B,2H]
        else:
            m = mask.unsqueeze(-1)                              # [T,B,1]
            h = (y*m).sum(0) / (m.sum(0).clamp_min(1.0))        # masked mean
        pair_logits = self.pair_head(h)                         # [B,10]
        return phase, digit_logits, pair_logits

# ============== Losses ==============
class PhaseLoss(nn.Module):
    def __init__(self, beta=0.1, lam_tv=0.1, lam_mono=0.2):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=beta)
        self.lam_tv=lam_tv; self.lam_mono=lam_mono

    def forward(self, pred, y, m):
        Lp = (self.huber(pred,y)*m).sum() / m.sum().clamp_min(1.0)
        valid = m[1:]*m[:-1]
        if valid.sum()>0:
            d = pred[1:]-pred[:-1]
            Ltv   = (d.abs()*valid).sum()/valid.sum()
            Lmono = (F.relu(-d)*valid).sum()/valid.sum()
        else:
            Ltv = pred.new_tensor(0.0); Lmono = pred.new_tensor(0.0)
        L = Lp + self.lam_tv*Ltv + self.lam_mono*Lmono
        return L, {"L_phase": float(Lp.detach()), "L_tv": float(Ltv.detach()), "L_mono": float(Lmono.detach())}

# ============== Train/Eval ==============
def train_one_epoch(model, loaders, opt, weights, device):
    model.train()
    phase_loss = PhaseLoss(beta=0.1, lam_tv=weights['lam_tv'], lam_mono=weights['lam_mono'])
    ce_digit = nn.CrossEntropyLoss(ignore_index=-100)
    ce_pair  = nn.CrossEntropyLoss()
    tr_loader = loaders['train']

    logs={"L":0,"L_phase":0,"L_digit":0,"L_pair":0,"L_tv":0,"L_mono":0}; n=0
    for x, yph, ycls, m, pid, _ in tr_loader:
        x,yph,ycls,m,pid = x.to(device), yph.to(device), ycls.to(device), m.to(device), pid.to(device)
        opt.zero_grad(set_to_none=True)
        ph, dlog, plog = model(x, mask=m)

        Lp, st = phase_loss(ph, yph, m)
        # digit
        T,B = dlog.size(0), dlog.size(1)
        dlog_flat = dlog.view(T*B, -1)
        ycls_flat = ycls.view(T*B)
        Ld = ce_digit(dlog_flat, ycls_flat)
        # pair
        Lc = ce_pair(plog, pid)

        L = weights['w_phase']*Lp + weights['w_digit']*Ld + weights['w_pair']*Lc
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        logs["L"]+=float(L.detach()); logs["L_phase"]+=st["L_phase"]; logs["L_tv"]+=st["L_tv"]; logs["L_mono"]+=st["L_mono"]
        logs["L_digit"]+=float(Ld.detach()); logs["L_pair"]+=float(Lc.detach()); n+=1
    for k in logs: logs[k]/=max(1,n)
    return logs

@torch.no_grad()
def eval_one_epoch(model, loaders, weights, device):
    model.eval()
    phase_loss = PhaseLoss(beta=0.1, lam_tv=weights['lam_tv'], lam_mono=weights['lam_mono'])
    ce_digit = nn.CrossEntropyLoss(ignore_index=-100)
    ce_pair  = nn.CrossEntropyLoss()
    va_loader = loaders['val']

    logs={"L":0,"L_phase":0,"L_digit":0,"L_pair":0,"L_tv":0,"L_mono":0}; n=0
    for x, yph, ycls, m, pid, _ in va_loader:
        x,yph,ycls,m,pid = x.to(device), yph.to(device), ycls.to(device), m.to(device), pid.to(device)
        ph, dlog, plog = model(x, mask=m)

        Lp, st = phase_loss(ph, yph, m)
        T,B = dlog.size(0), dlog.size(1)
        dlog_flat = dlog.view(T*B, -1)
        ycls_flat = ycls.view(T*B)
        Ld = ce_digit(dlog_flat, ycls_flat)
        Lc = nn.functional.cross_entropy(plog, pid)

        L = weights['w_phase']*Lp + weights['w_digit']*Ld + weights['w_pair']*Lc

        logs["L"]+=float(L.detach()); logs["L_phase"]+=st["L_phase"]; logs["L_tv"]+=st["L_tv"]; logs["L_mono"]+=st["L_mono"]
        logs["L_digit"]+=float(Ld.detach()); logs["L_pair"]+=float(Lc.detach()); n+=1
    for k in logs: logs[k]/=max(1,n)
    return logs

def split_dataset(ds, val_split=0.2):
    N=len(ds); n_val=max(1,int(round(N*val_split))); n_tr=N-n_val
    return torch.utils.data.random_split(ds, [n_tr, n_val])

def main():
    ap=argparse.ArgumentParser("Train dual-head Digit(0..9+transition) + Phase + Pair(10)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=20)
    ap.add_argument("--keep_ratio", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--low", type=float, default=0.4)
    ap.add_argument("--high", type=float, default=0.6)
    ap.add_argument("--w_phase", type=float, default=1.0)
    ap.add_argument("--w_digit", type=float, default=1.0)
    ap.add_argument("--w_pair", type=float, default=0.5)
    ap.add_argument("--lam_tv", type=float, default=0.1)
    ap.add_argument("--lam_mono", type=float, default=0.2)
    ap.add_argument("--save", type=str, default="digit_phase.pt")
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    full=DigitPhaseDataset(args.root, img_h=args.img_h, img_w=args.img_w, keep_ratio=args.keep_ratio, low=args.low, high=args.high)
    tr_set, va_set = split_dataset(full, args.val_split)
    tr_loader=DataLoader(tr_set,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,collate_fn=pad_collate,pin_memory=(device.type=='cuda'))
    va_loader=DataLoader(va_set,batch_size=args.batch,shuffle=False,num_workers=args.num_workers,collate_fn=pad_collate,pin_memory=(device.type=='cuda'))
    loaders={"train":tr_loader,"val":va_loader}

    model = CNNBiLSTM_DigitPhase(feat_dim=128, lstm_hidden=128, lstm_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights={"w_phase":args.w_phase,"w_digit":args.w_digit,"w_pair":args.w_pair,"lam_tv":args.lam_tv,"lam_mono":args.lam_mono}
    best=float('inf')
    for ep in range(1, args.epochs+1):
        tr=train_one_epoch(model, loaders, opt, weights, device)
        va=eval_one_epoch(model, loaders, weights, device)
        print(f"Epoch {ep:03d} | Ltr={tr['L']:.4f} (phase {tr['L_phase']:.4f}, digit {tr['L_digit']:.4f}, pair {tr['L_pair']:.4f}) | "
              f"Lval={va['L']:.4f}")
        if va["L"]<best:
            best=va["L"]
            torch.save({"model":model.state_dict(),"args":vars(args)}, args.save)
            print(f"  ✔ Save best to {args.save}")
    print("Done.")

if __name__=="__main__":
    main()
