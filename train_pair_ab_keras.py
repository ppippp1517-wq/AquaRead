# -*- coding: utf-8 -*-
"""
Train CNN for digit-transition (pair a->b) with sequence-wise validation,
and save charts/reports (loss/acc curves, confusion matrix, classification report, history.csv).

Python + TensorFlow/Keras
"""

import os, re, glob, math, random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical

# ================== PATHS ==================
DATA_ROOT        = r"D:\projectCPE\transition\dataset"      # โฟลเดอร์ 0_to_1 ... 9_to_0 ซึ่งมี seq_00001,...
MODEL_SAVE_PATH  = r"D:\projectCPE\pair_ab_keras.h5"        # ที่เก็บโมเดล
OUT_DIR          = r"D:\projectCPE\reports\pair_ab"         # ที่เก็บกราฟ/รายงาน
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ================== CONFIG =================
IMG_H, IMG_W = 32, 20          # input (H, W)
CHANNELS     = 1               # 1=gray, 3=RGB
VAL_RATIO    = 0.2             # แบ่งวาลแบบรายซีเควนซ์
SEED         = 42
EPOCHS       = 60
BATCH        = 16
LR           = 1e-3
AUG_LIGHT    = True            # เปิด augmentation เบา ๆ

# ================== UTILS ==================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def natural_key(p):
    base = os.path.basename(p)
    nums = re.findall(r'\d+', base)
    return [int(n) for n in nums] if nums else [base]

def resize_keep(im: Image.Image, W:int, H:int):
    w, h = im.size
    s = min(W / max(1, w), H / max(1, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new(im.mode, (W, H), color=0)
    off = ((W - nw) // 2, (H - nh) // 2)
    canvas.paste(im, off)
    return canvas

def load_image(path, to_rgb=False):
    im = Image.open(path).convert("RGB" if to_rgb else "L")
    im = resize_keep(im, IMG_W, IMG_H)
    arr = np.array(im, dtype=np.float32)
    if CHANNELS == 1:
        arr = arr[..., np.newaxis]  # (H,W,1)
    return arr

def simple_augment(x):
    # x: (H,W,1) [0..255]
    if not AUG_LIGHT:
        return x
    img = x.copy()
    # brightness
    if random.random() < 0.5:
        img = np.clip(img + (random.random()*64 - 32.0), 0, 255)
    # vertical shift
    if random.random() < 0.5:
        dy = random.randint(-2, 2)
        img = np.roll(img, dy, axis=0)
    # tiny zoom
    if random.random() < 0.3:
        scale = 1.0 + (random.random()*0.2 - 0.1)  # 0.9..1.1
        nh, nw = max(1, int(round(IMG_H*scale))), max(1, int(round(IMG_W*scale)))
        im = Image.fromarray(img.squeeze(-1).astype(np.uint8))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (IMG_W, IMG_H), color=0)
        off = ((IMG_W - nw)//2, (IMG_H - nh)//2)
        canvas.paste(im, off)
        img = np.array(canvas, dtype=np.float32)[..., np.newaxis]
    return img

# parse pair id from folder name: 0_to_1 -> 0, 9_to_0 -> 9
def parse_pair_id(dirname):
    m = re.search(r'([0-9])_to_([0-9])', dirname.replace('\\','/'))
    if not m: 
        return None
    a, b = int(m.group(1)), int(m.group(2))
    assert b == (a+1) % 10, f"Folder not a->a+1: {dirname}"
    return a

def collect_sequences(data_root):
    """return list of (seq_dir, pair_id)"""
    pair_dirs = sorted(glob.glob(os.path.join(data_root, "*_to_*")))
    seqs = []
    for pd in pair_dirs:
        a = parse_pair_id(pd)
        if a is None: 
            continue
        # seq folders inside
        sds = [d for d in glob.glob(os.path.join(pd, "seq_*")) if os.path.isdir(d)]
        sds.sort()
        for sd in sds:
            seqs.append((sd, a))
    return seqs

def split_sequences_stratified(seq_list, val_ratio=0.2, seed=42):
    """split by sequence for each class; if a class has <=1 sequence, keep it in train"""
    from collections import defaultdict
    groups = defaultdict(list)
    for sd, a in seq_list:
        groups[a].append(sd)
    rng = random.Random(seed)
    tr, va = [], []
    for a, sds in groups.items():
        rng.shuffle(sds)
        if len(sds) <= 1:
            tr += [(sd, a) for sd in sds]
        else:
            n_val = max(1, int(round(len(sds) * val_ratio)))
            va += [(sd, a) for sd in sds[:n_val]]
            tr += [(sd, a) for sd in sds[n_val:]]
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

def load_split_from_sequences(seq_list, augment=False):
    """load all frames from sequence dirs -> X, y, paths"""
    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
    X, Y, P = [], [], []
    for sd, a in seq_list:
        files = [os.path.join(sd, f) for f in os.listdir(sd)
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
        files.sort(key=natural_key)
        for p in files:
            arr = load_image(p, to_rgb=(CHANNELS==3))
            if augment:
                arr = simple_augment(arr)
            X.append(arr); Y.append(a); P.append(p)
    if len(X) == 0:
        return None, None, None
    X = np.stack(X, axis=0).astype(np.float32)  # (N,H,W,C)
    Y = to_categorical(np.array(Y), 10)         # 10 classes
    return X, Y, P

# ================== MODEL ===================
def build_model():
    model = Sequential(name="PairAB_Keras")
    model.add(BatchNormalization(input_shape=(IMG_H, IMG_W, CHANNELS)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation="softmax"))
    return model

# ================== TRAIN ===================
def main():
    set_seed(SEED)
    print("Scanning sequences in:", DATA_ROOT)
    seqs = collect_sequences(DATA_ROOT)
    # fallback: ถ้ายังไม่มี seq_* แต่มีไฟล์ในโฟลเดอร์ระดับบน ให้รองรับแบบแบน
    if len(seqs) == 0:
        print("WARNING: no seq_* found. Scanning one-level folders with images.")
        pair_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*_to_*")))
        for pd in pair_dirs:
            a = parse_pair_id(pd)
            if a is None: 
                continue
            seqs.append((pd, a))

    from collections import Counter
    c = Counter([a for _, a in seqs])
    print(f"Total sequences: {len(seqs)} (per-class: {c})")

    tr_seqs, va_seqs = split_sequences_stratified(seqs, VAL_RATIO, SEED)
    print(f"Train seq: {len(tr_seqs)} | Val seq: {len(va_seqs)}")

    X_train, y_train, P_train = load_split_from_sequences(tr_seqs, augment=True)
    X_val,   y_val,   P_val   = load_split_from_sequences(va_seqs, augment=False)

    if X_train is None:
        raise RuntimeError("No training frames found. Check dataset structure.")

    print(f"Train frames: {X_train.shape[0]} | Val frames: {0 if X_val is None else X_val.shape[0]}")
    # normalize to [0,1]
    X_train = X_train / 255.0
    if X_val is not None:
        X_val = X_val / 255.0

    model = build_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    callbacks = []
    ckpt_path = os.path.join(OUT_DIR, "best_pair_ab.keras")  # Keras format
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False
    ))
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    ))

    if X_val is None:
        history = model.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCHS, shuffle=True, verbose=1)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            batch_size=BATCH, epochs=EPOCHS, shuffle=True,
                            callbacks=callbacks, verbose=1)

    # save final model (H5)
    model.save(MODEL_SAVE_PATH)
    print("Saved model:", MODEL_SAVE_PATH)

    # ======== Evaluate on TRAIN (memorization check) ========
    print("\n=== Evaluate on TRAIN set (memorization check) ===")
    pred_train = model.predict(X_train, batch_size=64, verbose=0)
    pred_lbl = np.argmax(pred_train, axis=1)
    true_lbl = np.argmax(y_train, axis=1)
    acc_train = (pred_lbl == true_lbl).mean()
    print(f"Train accuracy = {acc_train*100:.2f}%  ({pred_lbl.size} samples)")

    # ======== Evaluate on VAL ========
    if X_val is not None:
        print("\n=== Evaluate on VALIDATION set ===")
        pred_val = model.predict(X_val, batch_size=64, verbose=0)
        pred_lbl_v = np.argmax(pred_val, axis=1)
        true_lbl_v = np.argmax(y_val, axis=1)
        acc_val = (pred_lbl_v == true_lbl_v).mean()
        print(f"Val accuracy = {acc_val*100:.2f}%  ({pred_lbl_v.size} samples)")

    # ======== Save charts & reports ========
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    # 1) history.csv
    pd.DataFrame(history.history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)

    # pick metric keys across TF versions
    acc_key = "accuracy" if "accuracy" in history.history else "categorical_accuracy"
    val_acc_key = f"val_{acc_key}"

    # 2) loss.png
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.epoch, history.history["val_loss"], label="val")
    plt.title("Model loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
    plt.grid(True, ls="--", alpha=.3); plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3) accuracy.png
    plt.figure()
    plt.plot(history.epoch, history.history[acc_key], label="train")
    if val_acc_key in history.history:
        plt.plot(history.epoch, history.history[val_acc_key], label="val")
    plt.title("Model accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.grid(True, ls="--", alpha=.3); plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4) confusion_matrix + classification_report
    if X_val is not None:
        y_true = true_lbl_v
        y_pred = pred_lbl_v
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        np.savetxt(os.path.join(OUT_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

        labels = [f"{i}->{(i+1)%10}" for i in range(10)]
        plt.figure(figsize=(6.2, 6.2))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix (Validation)")
        plt.xticks(range(10), labels, rotation=45, ha="right")
        plt.yticks(range(10), labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
        plt.close()

        report = classification_report(y_true, y_pred, target_names=labels,
                                       digits=4, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(OUT_DIR, "classification_report.csv"))

    print(f"Saved charts & reports to: {OUT_DIR}")

if __name__ == "__main__":
    main()
