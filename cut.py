# d:/projectCPE/cut.py
import argparse, os
from pathlib import Path
import cv2

# นำฟังก์ชันตัดกรอบด้านในที่ให้ไปก่อนหน้า
from crop_panel_inner import crop_panel_inner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True,
                    help="พาธของไฟล์ภาพ หรือ โฟลเดอร์รูป")
    ap.add_argument("--out", dest="outdir", required=True,
                    help="โฟลเดอร์เอาท์พุตสำหรับรูปที่ครอปแล้ว/ดีบัก")
    ap.add_argument("--inset", type=int, default=3,
                    help="ขยับเข้าจากขอบกรอบ (px) เพื่อตัดเส้นดำ")
    ap.add_argument("--thr", type=float, default=0.22,
                    help="threshold ratio ของสัญญาณขอบ (0..1)")
    ap.add_argument("--minrun", type=float, default=0.05,
                    help="สัดส่วน run length ขั้นต่ำของขอบ (0..1)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # สร้างรายชื่อไฟล์
    if inp.is_file():
        files = [inp]
    else:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = sorted([p for p in inp.iterdir() if p.suffix.lower() in exts])

    if not files:
        print(f"[ERR] ไม่พบไฟล์รูปใน: {inp}")
        return

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] อ่านรูปไม่เข้า: {p}")
            continue

        cropped, box, dbg = crop_panel_inner(
            img, inset_px=args.inset, thr_ratio=args.thr, min_run_ratio=args.minrun
        )
        # เซฟผล
        cv2.imwrite(str(outdir / f"{p.stem}_inner.png"), cropped)
        cv2.imwrite(str(outdir / f"{p.stem}_inner_dbg.png"), dbg)
        print(f"[OK] {p.name} -> box={box}")

if __name__ == "__main__":
    main()
