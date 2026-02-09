import os, re, shutil, argparse, pathlib
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.PNG','.JPG','.JPEG','.BMP'}

def extract_numbers(name: str):
    # ดึงเลขทุกชุดในชื่อ (ไม่รวมสกุล)
    nums = re.findall(r'\d+', name)
    return [int(x) for x in nums]

def hardlink_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists(): dst.unlink()
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='โฟลเดอร์ที่มีไฟล์แบน เช่น D:/projectCPE/pcic/0to1')
    ap.add_argument('--dst', required=True, help='ปลายทาง เช่น D:/projectCPE/transition/dataset/0_to_1')
    ap.add_argument('--dry_run', action='store_true', help='พรีวิวการแมปโดยไม่สร้างไฟล์')
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix in IMG_EXTS]
    if not files:
        raise SystemExit("ไม่พบไฟล์รูปในโฟลเดอร์ต้นทาง")

    groups = {}
    unmatched = []
    for p in files:
        nums = extract_numbers(p.stem)
        if len(nums) >= 2:
            seq, frame = nums[0], nums[-1]
        elif len(nums) == 1:
            # ทั้งโฟลเดอร์เป็น sequence เดียว
            seq, frame = 0, nums[0]
        else:
            unmatched.append(p.name)
            continue
        groups.setdefault(seq, []).append((frame, p))

    if not groups:
        raise SystemExit("ยังจับกลุ่มไม่ได้เลย (ไม่มีเลขในชื่อไฟล์)")

    # พรีวิว
    if args.dry_run:
        print(f"พบ {len(groups)} sequences")
        for k in sorted(groups)[:5]:
            ex = ', '.join([f"{frm}:{pp.name}" for frm, pp in sorted(groups[k])[:5]])
            print(f"  seq {k:05d} -> {len(groups[k])} frames | ตัวอย่าง: {ex}")
        if unmatched:
            print(f"\nตัวอย่างชื่อที่ไม่ match ({min(10, len(unmatched))}):")
            for n in unmatched[:10]:
                print("  -", n)
        return

    # สร้าง hardlink/copy
    total = 0
    for seq, lst in sorted(groups.items()):
        lst.sort(key=lambda x: (x[0], x[1].name))
        seq_dir = dst / f"seq_{seq:05d}"
        for frm, src_path in lst:
            dst_path = seq_dir / f"{frm:03d}{src_path.suffix.lower()}"
            hardlink_or_copy(src_path, dst_path)
            total += 1

    print(f"✔ จัดกลุ่มเสร็จ: {len(groups)} sequences, รวม {total} รูป")
    if unmatched:
        print(f"(ข้ามชื่อที่ไม่ match {len(unmatched)} ไฟล์)")
    print(f"ไปที่: {dst}")

if __name__ == '__main__':
    main()
