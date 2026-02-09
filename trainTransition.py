# trainTransition.py  (drop-in)
import os, re, glob, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# ===== CONFIG =====
DATA_ROOT   = r"D:\projectCPE\transition\dataset"
SAVE_DIR    = r"D:\projectCPE\transition\prepared_ds"
MODEL_PATH  = r"D:\projectCPE\transition\transition_cnn_lstm.keras"

IMG_W, IMG_H, CHANNELS = 20, 32, 3
SEQ_LEN = 3
USE_TRANSITION_CLASS = True        # ตั้ง False ถ้าอยากเริ่มแบบไม่มีคลาส 10
USE_TRANSITION_CLASS = False
NUM_CLASSES = 10


VAL_SPLIT = 0.15
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
AUG_ON = True

TIMESTEP_WEIGHTS = tf.constant([2.0, 1.0, 2.0], dtype=tf.float32)
TRANSITION_WEIGHT = 0.5
DIGIT_WEIGHT = 2.0

os.makedirs(SAVE_DIR, exist_ok=True)

PAIR_DIR = re.compile(r"([0-9])_to_([0-9])$", re.IGNORECASE)

def find_frame(seq_dir, idx):
    # หาไฟล์ 000.*, 001.*, 002.* (รองรับนามสกุลทุกแบบ)
    pats = glob.glob(os.path.join(seq_dir, f"{idx:03d}.*"))
    return pats[0] if pats else None

def load_seq_paths(root):
    items = []
    pair_dirs = sorted(glob.glob(os.path.join(root, "*_to_*")))
    print(f"[INFO] Found pair folders: {len(pair_dirs)} at {root}")
    for pair_dir in pair_dirs:
        m = PAIR_DIR.search(os.path.basename(pair_dir))
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        seq_dirs = sorted(glob.glob(os.path.join(pair_dir, "seq_*")))
        for sd in seq_dirs:
            f0, f1, f2 = find_frame(sd, 0), find_frame(sd, 1), find_frame(sd, 2)
            if all([f0, f1, f2]):
                items.append((sd, a, b, [f0, f1, f2]))
            else:
                # debug รายการที่ขาดไฟล์
                missing = [i for i, f in enumerate([f0, f1, f2]) if f is None]
                print(f"[WARN] skip {sd}: missing frames {missing}")
    print(f"[INFO] Total sequences found: {len(items)}")
    # โชว์ตัวอย่าง 3 อันแรก
    for ex in items[:3]:
        print("[SAMPLE]", ex[0], "->", ex[2], "frames:", [os.path.basename(p) for p in ex[3]])
    return items

def read_img(path):
    img = cv2.imread(path)  # BGR
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def labels_for_sequence(a, b):
    return np.array([a, TRANSITION_ID, b], np.int64) if USE_TRANSITION_CLASS else np.array([a, b, b], np.int64)

def build_arrays(seq_items):
    X_list, y_list = [], []
    for seq_dir, a, b, fps in seq_items:
        imgs = [read_img(p) for p in fps]
        X_list.append(np.stack(imgs, axis=0))  # (L,H,W,C)
        y_list.append(labels_for_sequence(a, b))
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y

def aug_one(img):
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    pad = 1
    img = tf.pad(img, [[pad,pad],[pad,pad],[0,0]], mode='REFLECT')
    img = tf.image.random_crop(img, size=[IMG_H, IMG_W, CHANNELS])
    return tf.clip_by_value(img, 0.0, 1.0)

def make_ds(X, y, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_train: ds = ds.shuffle(min(4096, len(X)))

    def _map(a, b):
        if is_train and AUG_ON:
            a = tf.map_fn(aug_one, a, fn_output_signature=tf.float32)
        if USE_TRANSITION_CLASS:
            class_w = tf.where(tf.equal(b, TRANSITION_ID), TRANSITION_WEIGHT, DIGIT_WEIGHT)
        else:
            class_w = tf.fill(tf.shape(b), DIGIT_WEIGHT)
        class_w = tf.cast(class_w, tf.float32)[..., None]
        sw = class_w * TIMESTEP_WEIGHTS[..., None]
        return a, tf.cast(b[..., None], tf.int32), sw

    return ds.map(_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model():
    cnn = models.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, CHANNELS)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
    ], name="cnn_extractor")

    inp = layers.Input(shape=(SEQ_LEN, IMG_H, IMG_W, CHANNELS))
    x = layers.TimeDistributed(cnn)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation='softmax'))(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    seqs = load_seq_paths(DATA_ROOT)
    if not seqs:
        print("[ERROR] ไม่พบ sequence ที่ครบ 000/001/002.* เลย — ตรวจพาธ/ชื่อไฟล์อีกครั้ง")
        return

    random.shuffle(seqs)
    n_total = len(seqs)
    n_val = max(1, int(n_total * VAL_SPLIT))
    val_items, train_items = seqs[:n_val], seqs[n_val:]

    X_tr, y_tr = build_arrays(train_items)
    X_va, y_va = build_arrays(val_items)
    np.save(os.path.join(SAVE_DIR, "X_tr.npy"), X_tr)
    np.save(os.path.join(SAVE_DIR, "y_tr.npy"), y_tr)
    np.save(os.path.join(SAVE_DIR, "X_va.npy"), X_va)
    np.save(os.path.join(SAVE_DIR, "y_va.npy"), y_va)
    print("Train:", X_tr.shape, y_tr.shape, " Val:", X_va.shape, y_va.shape)
    print("Mode:", "WITH transition class (10)" if USE_TRANSITION_CLASS else "NO transition class (middle=b)")

    train_ds = make_ds(X_tr, y_tr, True)
    val_ds   = make_ds(X_va, y_va, False)

    model = build_model()
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, es])
    print("Saved best model to:", MODEL_PATH)

    if val_items:
        sd, a, b, fps = val_items[0]
        X = np.expand_dims(np.stack([read_img(p) for p in fps], axis=0), 0)
        pred = np.argmax(model.predict(X), axis=-1)[0]
        print("Val sample:", sd)
        print("Ground truth:", (np.array([a, TRANSITION_ID, b]) if USE_TRANSITION_CLASS else np.array([a,b,b])).tolist())
        print("Predicted   :", pred.tolist())

if __name__ == "__main__":
    main()
