# check_dataset_digits.py
import os, glob, csv
from collections import Counter, defaultdict

# ================== CONFIG ==================
DATASET_ROOT = r"D:/projectCPE/dataset_digital"
N_CLASSES = 5                                # digital0..digital4 => 0..4
IMAGE_DIR = "images"                         # images/train, images/val
LABEL_DIR = "labels"                         # labels/train, labels/val
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ISSUE_CSV = "issues_report.csv"
# ============================================

def read_label_file(path, n_classes):
    """return: rows=[(cls,cx,cy,w,h), ...], issues=[str,...]"""
    issues, rows = [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return rows, ["missing_label_file"]

    for i, ln in enumerate(lines, 1):
        parts = ln.split()
        if len(parts) < 5:
            issues.append(f"line{i}_format_error")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except Exception:
            issues.append(f"line{i}_parse_error")
            continue

        if not (0 <= cls < n_classes):
            issues.append(f"line{i}_class_out_of_range({cls})")

        for k, v in zip(("cx","cy","w","h"), (cx,cy,w,h)):
            if not (0 <= v <= 1):
                issues.append(f"line{i}_{k}_out_of_[0,1]({v:.4f})")

        if w <= 0 or h <= 0:
            issues.append(f"line{i}_zero_or_negative_wh(w={w:.4f},h={h:.4f})")
        if w < 0.005 or h < 0.005:
            issues.append(f"line{i}_very_small_wh(w={w:.4f},h={h:.4f})")

        rows.append((cls, cx, cy, w, h))
    return rows, issues

def scan_split(split):
    img_dir = os.path.join(DATASET_ROOT, IMAGE_DIR, split)
    lbl_dir = os.path.join(DATASET_ROOT, LABEL_DIR, split)
    images = [p for p in glob.glob(os.path.join(img_dir, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]
    images.sort()

    per_class = Counter()
    per_file_counts = {}
    problems = []  # list of dict rows for CSV

    missing_label_files = []
    missing_images = []  # label exists but image missing
    label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    image_basenames = {os.path.splitext(os.path.basename(p))[0] for p in images}
    label_basenames = {os.path.splitext(os.path.basename(p))[0] for p in label_files}

    for base in sorted(label_basenames - image_basenames):
        problems.append({"split": split, "image": base, "issue": "label_has_no_matching_image", "detail": ""})
        missing_images.append(base)

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        rows, issues = read_label_file(lbl_path, N_CLASSES)

        if not rows and "missing_label_file" in issues:
            problems.append({"split": split, "image": base, "issue": "missing_label_file", "detail": ""})
            missing_label_files.append(base)
            continue

        # count classes
        cls_ids = [r[0] for r in rows]
        for c in cls_ids:
            if 0 <= c < N_CLASSES:
                per_class[c] += 1
        per_file_counts[base] = Counter(cls_ids)

        # expected: exactly one of each 0..4
        expected = set(range(N_CLASSES))
        present = {c for c in cls_ids if 0 <= c < N_CLASSES}
        missing_ids = sorted(list(expected - present))
        extra_ids   = sorted([c for c in present if cls_ids.count(c) > 1])
        if missing_ids:
            problems.append({"split": split, "image": base, "issue": "missing_class_ids", "detail": str(missing_ids)})
        if extra_ids:
            problems.append({"split": split, "image": base, "issue": "duplicate_class_ids", "detail": str(extra_ids)})

        # record line-level issues
        for it in issues:
            problems.append({"split": split, "image": base, "issue": it, "detail": ""})

        # sanity: total lines != 5
        if len([r for r in rows if 0 <= r[0] < N_CLASSES]) != N_CLASSES:
            problems.append({"split": split, "image": base, "issue": "not_exactly_5_yolo_lines", "detail": f"{len(rows)}"})

    return {
        "num_images": len(images),
        "per_class": per_class,
        "per_file_counts": per_file_counts,
        "problems": problems,
        "missing_label_files": missing_label_files,
        "missing_images": missing_images,
    }

def main():
    all_problems = []
    summary = {}

    for split in ("train","val"):
        res = scan_split(split)
        summary[split] = res

        print(f"\n=== [{split.upper()}] ===")
        print(f"images: {res['num_images']}")
        # class counts
        row = " | ".join([f"class{c}:{res['per_class'].get(c,0)}" for c in range(N_CLASSES)])
        print("label counts ->", row)
        print(f"missing label files: {len(res['missing_label_files'])}, labels without image: {len(res['missing_images'])}")
        # show some problematic files
        counts = defaultdict(int)
        for p in res["problems"]:
            counts[p["issue"]] += 1
        if counts:
            print("issues summary ->", dict(sorted(counts.items(), key=lambda x: -x[1])[:10]))
        else:
            print("issues summary -> NONE")
        all_problems.extend(res["problems"])

    # write CSV report
    if all_problems:
        with open(ISSUE_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["split","image","issue","detail"])
            w.writeheader()
            w.writerows(all_problems)
        print(f"\nCSV report written: {ISSUE_CSV}  (rows={len(all_problems)})")
    else:
        print("\nNo issues found ✅")

if __name__ == "__main__":
    main()
