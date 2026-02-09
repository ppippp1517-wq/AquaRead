import os, glob, re, math, shutil

DATA_ROOT = r"D:\projectCPE\transition\dataset"  # โฟลเดอร์ 0_to_1 ... 9_to_0
# วิธีเลือกอย่างใดอย่างหนึ่ง:
CHUNK_SIZE = 80        # จำนวนภาพต่อซีเควนซ์ (เช่น 80 รูป / seq)
# หรือกำหนดจำนวนซีเควนซ์ที่อยากได้ (ยกเลิก CHUNK_SIZE โดยตั้ง N_SEQ > 0)
N_SEQ = 0              # เช่น 5  (ถ้าตั้งค่านี้ >0 จะกระจายภาพให้ได้ N ซีเควนซ์เท่า ๆ กัน)

IMG_EXT = {".png",".jpg",".jpeg",".bmp"}

def natural_key(p):
    import re, os
    b = os.path.basename(p); nums = re.findall(r'\d+', b)
    return [int(n) for n in nums] if nums else [b]

for cls in sorted(os.listdir(DATA_ROOT)):
    cdir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cdir): 
        continue
    files = [os.path.join(cdir,f) for f in os.listdir(cdir)
             if os.path.splitext(f)[1].lower() in IMG_EXT]
    files.sort(key=natural_key)

    if not files:
        continue

    # ตัดสินใจแบ่งอย่างไร
    if N_SEQ and N_SEQ > 0:
        per = math.ceil(len(files)/N_SEQ)
    else:
        per = max(1, CHUNK_SIZE)

    # ลบโฟลเดอร์ seq_* เดิมก่อน (กันซ้ำ)
    for d in glob.glob(os.path.join(cdir, "seq_*")):
        if os.path.isdir(d):
            shutil.rmtree(d)

    # แบ่งและย้ายไฟล์
    seq_idx = 0
    for i in range(0, len(files), per):
        seq_idx += 1
        out = os.path.join(cdir, f"seq_{seq_idx:05d}")
        os.makedirs(out, exist_ok=True)
        for p in files[i:i+per]:
            shutil.move(p, os.path.join(out, os.path.basename(p)))
    print(f"{cls}: {len(files)} imgs -> {seq_idx} sequences")
