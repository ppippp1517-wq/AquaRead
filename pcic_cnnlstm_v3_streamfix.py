
"""
PCIC CNN+BiLSTM v3 (stream-fix)
- Sequence-level split (same as v3).
- Robust streaming gate:
    Accept digit if (phase >= threshold) OR (digit stable >= K frames AND confidence >= p_th).
- Easy DECIMAL_SHIFT selector for the digit position you are training.
- Debug: print phase/pred/conf only when output value changes.

Usage:
    python pcic_cnnlstm_v3_streamfix.py
"""

import os, re, math, time, random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------ CONFIG ------------------
ROOT_DIR      = r"D:\projectCPE\pcic"
IMG_SIZE      = (64, 64)
WINDOW_LEN    = 16
WINDOW_STRIDE = 4
BATCH_SIZE    = 16
EPOCHS        = 8          # keep short; we mainly use existing best ckpt
LR_MAX        = 2e-3
LAMBDA_PHASE  = 0.7
EDGE_LO       = 0.10
EDGE_HI       = 0.90
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 1234

# Streaming & value composing
DECIMAL_SHIFT = 0      # <<< set 0 for ones place; -1 for tenths; -3 for 0.001
PHASE_THRESH  = 0.80   # more permissive than 0.95
STABLE_K      = 3      # need same digit at least K windows
CONF_THRESH   = 0.80   # average softmax confidence threshold

CKPT_PATH     = "./checkpoints/pcic_cnnlstm_v3_best.pt"  # model to load for streaming

# ------------------ HELPERS ------------------
_digit_folder_re = re.compile(r"([0-9])\s*to\s*([0-9])", re.IGNORECASE)

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_image(path, size_hw=(64,64)):
    im = Image.open(path).convert("L")
    im = im.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return im

def _to_tensor_norm(im: Image.Image):
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1,1]
    arr = arr[None, ...]
    return torch.from_numpy(arr)

def _augment(im: Image.Image):
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
    def __init__(self, name:str, x_digit:int, y_digit:int, frame_paths:List[str]):
        self.name = name; self.x = x_digit; self.y = y_digit; self.frame_paths = frame_paths
    def __len__(self): return len(self.frame_paths)
    def frame_phase(self, idx:int) -> float:
        if len(self.frame_paths) <= 1: return 0.0
        return idx / (len(self.frame_paths)-1)

def discover_sequences(root_dir:str):
    seqs = []
    for name in sorted(os.listdir(root_dir)):
        m = _digit_folder_re.match(name)
        if not m: continue
        x = int(m.group(1)); y = int(m.group(2))
        folder = os.path.join(root_dir, name)
        if not os.path.isdir(folder): continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        if not files: continue
        files.sort()
        seqs.append(PCICSequence(name, x, y, files))
    print(f"[INFO] Found {len(seqs)} sequences:", [s.name for s in seqs])
    return seqs

# ------------------ MODEL ------------------
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
    def forward(self, x):
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
        digit_logits = self.head_digit(y)
        phase = torch.sigmoid(self.head_phase(y))
        return digit_logits, phase

# ------------------ STREAM READER ------------------
class StreamReader:
    def __init__(self, model, device, window_len=16, stride=2,
                 phase_threshold=0.8, decimal_shift=0,
                 allow_negative=False, max_rate=None,
                 stable_k=3, conf_thresh=0.8):
        self.model = model.eval()
        self.device = device
        self.buf = deque(maxlen=window_len)
        self.stride = stride
        self.decimal_shift = decimal_shift
        self.phase_threshold = phase_threshold
        self.allow_negative = allow_negative
        self.max_rate = max_rate
        self.pre_value = 0.0
        self.last_emit_t = time.time()
        # stability gate
        self.last_digit = None
        self.stable_k = 0
        self.stable_need = stable_k
        self.conf_thresh = conf_thresh
        self._last_print_val = None

    @torch.no_grad()
    def push_frame(self, arr_1xHxW: np.ndarray):
        self.buf.append(torch.from_numpy(arr_1xHxW).unsqueeze(0))  # (1,1,H,W)
        if len(self.buf) < self.buf.maxlen or (len(self.buf) % self.stride)!=0:
            return None

        clips = torch.stack(list(self.buf), dim=1).to(self.device)  # (1,T,1,H,W)
        digit_logits, phase_pred = self.model(clips)

        tail = 4
        d_tail = digit_logits[:, -tail:, :].softmax(-1)   # (1,tail,10)
        p_tail = phase_pred[:, -tail:, :]                 # (1,tail,1)

        # predicted digit & confidence
        probs = d_tail.squeeze(0)                         # (tail,10)
        confs, ids = probs.max(-1)                        # (tail,), (tail,)
        pred_digit = int(ids.mode()[0].item())
        conf_avg = float(confs.mean().item())
        phase = float(p_tail.mean().item())

        # stability tracking
        if self.last_digit is None or pred_digit != self.last_digit:
            self.last_digit = pred_digit; self.stable_k = 1
        else:
            self.stable_k += 1

        # gating: phase OR stability+confidence
        accept = (phase >= self.phase_threshold) or (self.stable_k >= self.stable_need and conf_avg >= self.conf_thresh)
        if accept:
            symbol = pred_digit
        else:
            symbol = None

        # compose numeric value
        unit = 10 ** (self.decimal_shift)
        if symbol is None:
            value = self.pre_value
        else:
            base = math.floor(self.pre_value / unit) * unit
            value = base + symbol * unit

        # emit only if changes
        if self._last_print_val is None or value != self._last_print_val:
            print(f"[stream] phase={phase:.2f} conf={conf_avg:.2f} pred={pred_digit} "
                  f"stable={self.stable_k} accept={accept} -> value={value}")
            self._last_print_val = value

        self.pre_value = value
        self.last_emit_t = time.time()
        return value

# ------------------ MAIN ------------------
def main():
    set_seed(SEED)
    # load best checkpoint
    model = CNNLSTM(in_ch=1, feat=128, hidden=128, num_layers=2, num_classes=10).to(DEVICE)
    if not os.path.exists(CKPT_PATH):
        print(f"[WARN] {CKPT_PATH} not found. Please train v3 first to create best checkpoint.")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    # pick a sequence to stream (first folder)
    seqs = []
    for name in sorted(os.listdir(ROOT_DIR)):
        m = _digit_folder_re.match(name)
        if not m: continue
        folder = os.path.join(ROOT_DIR, name)
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        if not files: continue
        files.sort()
        seqs.append((name, files))
    if not seqs:
        print("[ERR] no sequences found.")
        return

    name, files = seqs[0]
    print(f"[DEMO] Streaming over sequence: {name}  frames={len(files)}")

    def _to_arr_norm(path):
        im = _load_image(path, IMG_SIZE)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return arr[None, ...]

    sr = StreamReader(model, DEVICE, window_len=WINDOW_LEN, stride=2,
                      phase_threshold=PHASE_THRESH, decimal_shift=DECIMAL_SHIFT,
                      allow_negative=False, max_rate=None,
                      stable_k=STABLE_K, conf_thresh=CONF_THRESH)

    for p in files:
        sr.push_frame(_to_arr_norm(p))

if __name__ == "__main__":
    main()
