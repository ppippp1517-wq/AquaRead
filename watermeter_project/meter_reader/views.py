# meter_reader/views.py
import os, glob, shutil, base64, uuid, time
from datetime import datetime

from django.conf import settings
from django.http import (
    HttpResponse, JsonResponse, HttpResponseBadRequest, StreamingHttpResponse
)
from django.shortcuts import render, redirect
from django.utils.timezone import now
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.views import redirect_to_login
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from functools import wraps
import random

from .utils import (
    process_meter_image,
    SUMMARY_DIR, DETECT_DIGIT_DIR, CROP_CLASSX_DIR, DIGIT_CROPS_DIR,
)
from .camera import get_camera
from urllib.parse import unquote

# ===== mock รายชื่อ/ที่อยู่/ซีเรียล สำหรับสุ่มแสดงผล =====
MOCK_NAMES = [
    "คุณสมชาย ใจดี", "คุณสุดารัตน์", "คุณพรทิพย์", "โครงการบ้านธีรพล",
    "หมู่บ้านพฤกษา 5/12", "คุณอัญชลี", "คุณธนกฤต", "คุณศุภกานต์"
]
MOCK_HOUSES = ["88/12", "88/13", "21/7", "12/21", "57/19", "99/8", "45/3", "7/2"]
MOCK_SERIALS = ["SPP-NE-000101", "SPP-NE-000102", "SPP-NE-000103",
                "SPP-CE-000201", "SPP-SO-000301"]
MOCK_ZONE = "เขตเหนือ"
MOCK_BRANCHES = ["สาขา A (เหนือ)", "สาขา B (เหนือ)"]

def _mock_row(filename: str, total: float) -> dict:
    """คืนแถวข้อมูลจำลอง 1 แถว สำหรับใส่ในตารางหน้า app"""
    name = random.choice(MOCK_NAMES)
    house = random.choice(MOCK_HOUSES)
    serial = random.choice(MOCK_SERIALS)
    branch = random.choice(MOCK_BRANCHES)
    conf = random.choice([0.91, 0.84, 0.78, 0.63, 0.42])
    status = "ปกติ" if conf >= 0.8 else ("เฝ้าระวัง" if conf >= 0.6 else "หยุดส่ง")
    return {
        "filename": filename,
        "zone": MOCK_ZONE,
        "branch": branch,
        "house_no": house,
        "customer": name,
        "serial": serial,
        "latest": float(f"{total:.2f}"),
        "updated_at": now().strftime("%Y-%m-%d %H:%M"),
        "conf_pct": int(conf * 100),
        "status": status,
    }


# ---------- Guest / Auth Helpers ----------
def _to_media_url(src_path: str | None) -> str | None:
    """Copy ไฟล์ไป MEDIA/outputs แล้วคืน URL ให้ template ใช้แสดงรูป"""
    if not src_path or not os.path.isfile(src_path):
        return None
    dst = os.path.join(settings.MEDIA_ROOT, "outputs", os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isfile(dst):
        shutil.copy(src_path, dst)
    return settings.MEDIA_URL.rstrip('/') + "/outputs/" + os.path.basename(src_path)


def _is_guest(request) -> bool:
    return bool(request.session.get("guest"))


def login_or_guest(view_func):
    """อนุญาตผู้ใช้ที่ 'ล็อกอิน' หรือ 'โหมดใช้งานครั้งเดียว (guest)'"""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if request.user.is_authenticated or _is_guest(request):
            return view_func(request, *args, **kwargs)

        # ถ้ายังไม่ล็อกอิน และไม่ใช่ guest → เด้งไปหน้า login ของเราเอง
        login_url = reverse('login')  # ชี้ไป path("login/", ...) ที่คุณมี
        return redirect_to_login(
            request.get_full_path(),
            login_url=login_url,
        )

    return _wrapped


def app_entry(request):
    """
    จุดเข้าแอพจาก Android:
      - ถ้า login แล้ว หรือเป็น guest → ไปหน้า meter_app
      - ถ้ายังไม่ login และไม่ใช่ guest → ไปหน้า login
    """
    if request.user.is_authenticated or _is_guest(request):
        return redirect('meter_app')
    return redirect('login')


# ---------- Landing / Overview ----------
@never_cache
def overview(request):
    return render(request, "meter_reader/overview_public.html", {"active": "overview"})


def start_choice(request):
    return render(request, "meter_reader/start_choice.html", {"active": "overview"})


def start_guest(request):
    request.session["guest"] = True
    return redirect("meter_app")  # ไปหน้าใช้งานหลัก


# ---------- App main (guest or login) ----------
@login_or_guest
def app(request):
    guest = _is_guest(request) or (not request.user.is_authenticated)
    # ถ้าล็อกอินแล้วเคลียร์สถานะ guest
    if request.user.is_authenticated and request.session.get("guest"):
        request.session.pop("guest", None)

    recent_results = request.session.get("recent_results", [])
    latest_batch = request.session.get("latest_batch", [])  # ใช้เติมตารางในหน้า app

    return render(
        request,
        "meter_reader/app.html",
        {
            "active": "upload",
            "recent_results": recent_results,
            "latest_batch": latest_batch,
            "guest": guest,
        },
    )


# ---------- Simple index (legacy demo) ----------
def index(request):
    return render(request, "meter_reader/index.html", {"active": "upload"})


# ---------- API: analyze single file from <input>/<canvas> ----------
@csrf_exempt
@never_cache
def analyze(request):
    if request.method != "POST" or "file" not in request.FILES:
        return HttpResponseBadRequest('POST multipart/form-data with "file"')

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    fname = f'web_{uuid.uuid4().hex}.jpg'
    fpath = os.path.join(upload_dir, fname)

    with open(fpath, "wb+") as dst:
        for chunk in request.FILES["file"].chunks():
            dst.write(chunk)

    meter_id = request.POST.get("meter_id") or request.GET.get("meter_id") or "SPP-01"
    result = process_meter_image(fpath, meter_id=meter_id) or {}

    value = result.get("total") or result.get("value") or result.get("reading")
    boxes = result.get("boxes", [])

    overlay_path = result.get("detected_image_path")
    if overlay_path and overlay_path.startswith(settings.MEDIA_ROOT):
        overlay_url = settings.MEDIA_URL + os.path.relpath(overlay_path, settings.MEDIA_ROOT).replace("\\", "/")
    elif overlay_path:
        overlay_url = overlay_path
    else:
        overlay_url = settings.MEDIA_URL + f"uploads/{fname}"

    return JsonResponse({"value": value, "boxes": boxes, "overlay_url": overlay_url})


# ---------- Upload a whole folder ----------
# ===== mock รายชื่อ/ที่อยู่/ซีเรียล สำหรับสุ่มแสดงผล =====
MOCK_NAMES = [
    "คุณสมชาย ใจดี",
    "คุณสุดารัตน์",
    "คุณพรทิพย์",
    "โครงการบ้านธีรพล",
    "หมู่บ้านพฤกษา 5/12",
    "คุณอัญชลี",
    "คุณธนกฤต",
    "คุณศุภกานต์",
]
MOCK_HOUSES = ["88/12", "88/13", "21/7", "12/21", "57/19", "99/8", "45/3", "7/2"]
MOCK_SERIALS = [
    "SPP-NE-000101",
    "SPP-NE-000102",
    "SPP-NE-000103",
    "SPP-CE-000201",
    "SPP-SO-000301",
]
MOCK_ZONE = "เขตเหนือ"
MOCK_BRANCHES = ["สาขา A (เหนือ)", "สาขา B (เหนือ)"]


def _mock_pool_for_batch(n: int):
    """สุ่มชุดข้อมูลให้ครบก่อนหนึ่งครั้ง เพื่อลดการซ้ำใน batch"""
    names = random.sample(MOCK_NAMES, k=min(n, len(MOCK_NAMES)))
    houses = random.sample(MOCK_HOUSES, k=min(n, len(MOCK_HOUSES)))
    serials = random.sample(MOCK_SERIALS, k=min(n, len(MOCK_SERIALS)))
    branches = random.sample(MOCK_BRANCHES, k=min(n, len(MOCK_BRANCHES)))

    pool = []
    for i in range(n):
        pool.append(
            {
                "customer": names[i % len(names)],
                "house_no": houses[i % len(houses)],
                "serial": serials[i % len(serials)],
                "branch": branches[i % len(branches)],
            }
        )
    return pool


@login_or_guest
def upload_folder(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)

    uploaded_files = request.FILES.getlist("images")
    save_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_folder")
    os.makedirs(save_dir, exist_ok=True)

    results: list[dict] = []  # สำหรับหน้า folder_result.html
    app_rows: list[dict] = []  # สำหรับหน้า app.html ผ่าน session

    # สุ่มชุดข้อมูลไว้ล่วงหน้า ตามจำนวนไฟล์
    pool = _mock_pool_for_batch(len(uploaded_files))

    for idx, f in enumerate(uploaded_files):
        filename = os.path.basename(f.name)
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        meter_id = request.POST.get("meter_id", "SPP-01")
        res = process_meter_image(save_path, meter_id=meter_id) or {}

        if res.get("total") is not None:
            total = float(res["total"])

            base = pool[idx]
            conf = random.choice([0.91, 0.84, 0.78, 0.63, 0.42])
            status = "ปกติ" if conf >= 0.8 else ("เฝ้าระวัง" if conf >= 0.6 else "หยุดส่ง")

            # แถวเต็มสำหรับหน้า app
            mock = {
                "filename": filename,
                "zone": MOCK_ZONE,
                "branch": base["branch"],
                "house_no": base["house_no"],
                "customer": base["customer"],
                "serial": base["serial"],
                "latest": float(f"{total:.2f}"),
                "updated_at": now().strftime("%Y-%m-%d %H:%M"),
                "conf_pct": int(conf * 100),
                "status": status,
            }

            # สำหรับหน้า /upload_folder/
            results.append(
                {
                    "filename": filename,
                    "house_no": mock["house_no"],
                    "customer": mock["customer"],
                    "serial": mock["serial"],
                    "total": total,
                }
            )

            app_rows.append(mock)

    # เขียน CSV (มีคอลัมน์จำลอง)
    import csv

    csv_path = os.path.join(save_dir, "result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "house_no", "customer", "serial", "total"]
        )
        writer.writeheader()
        writer.writerows(results)

    # อัปเดตประวัติล่าสุดแบบสั้น
    history = request.session.get("recent_results", [])
    for row in results:
        history.insert(
            0,
            {
                "filename": row["filename"],
                "total": row["total"],
                "date": now().strftime("%Y-%m-%d %H:%M"),
            },
        )
    request.session["recent_results"] = history[:5]

    # เก็บชุดข้อมูลเต็มเพื่อไปโชว์ในหน้า app
    request.session["latest_batch"] = app_rows

    return render(
        request,
        "meter_reader/folder_result.html",
        {
            "active": "upload",
            "results": results,
            "csv_path": csv_path,
            "recent_results": request.session.get("recent_results", []),
            "guest": _is_guest(request) or (not request.user.is_authenticated),
        },
    )


# ---------- Webcam preview / confirm ----------
@csrf_exempt
@never_cache
@login_or_guest
def preview_image(request):
    cam = get_camera(index=0)
    jpeg = cam.get_jpeg()
    if not jpeg:
        return HttpResponse("ไม่พบภาพจากกล้อง USB", status=503)

    encoded = base64.b64encode(jpeg).decode("utf-8")
    request.session["preview_image_bytes"] = encoded

    return render(
        request,
        "meter_reader/preview.html",
        {
            "active": "upload",
            "base64_image": f"data:image/jpeg;base64,{encoded}",
            "guest": _is_guest(request) or (not request.user.is_authenticated),
        },
    )


@login_or_guest
def confirm_image(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)

    encoded = request.session.get("preview_image_bytes")
    if not encoded:
        return HttpResponse("ไม่มีภาพใน session", status=400)

    image_bytes = base64.b64decode(encoded)
    folder = os.path.join(settings.MEDIA_ROOT, "capture_images")
    os.makedirs(folder, exist_ok=True)
    filename = f"confirmed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    meter_id = request.POST.get("meter_id", "SPP-01")
    result = process_meter_image(path, meter_id=meter_id) or {}
    image_url = result.get("detected_image_path") or (
        settings.MEDIA_URL + f"capture_images/{filename}"
    )

    return render(
        request,
        "meter_reader/app.html",
        {
            "active": "upload",
            "result": result,
            "image_url": image_url,
            "recent_results": request.session.get("recent_results", []),
            "guest": _is_guest(request) or (not request.user.is_authenticated),
        },
    )


# ---------- One-shot capture & show ----------
@never_cache
@login_or_guest
def capture_image(request):
    cam = get_camera(index=0)
    jpeg = cam.get_jpeg()
    if not jpeg:
        return HttpResponse("ไม่พบภาพจากกล้อง USB", status=503)

    folder = os.path.join(settings.MEDIA_ROOT, "capture_images")
    os.makedirs(folder, exist_ok=True)
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(folder, filename)
    with open(image_path, "wb") as f:
        f.write(jpeg)

    image_url = settings.MEDIA_URL + f"capture_images/{filename}"
    return render(
        request,
        "meter_reader/app.html",
        {
            "active": "upload",
            "image_url": image_url,
            "recent_results": request.session.get("recent_results", []),
            "guest": _is_guest(request) or (not request.user.is_authenticated),
        },
    )

@login_or_guest
def upload_image(request):
    result, image_url = None, None
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        image_path = os.path.join(upload_dir, os.path.basename(image.name))
        with open(image_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        meter_id = request.POST.get("meter_id", "SPP-01")
        result = process_meter_image(image_path, meter_id=meter_id) or {}

        # URL ของรูปที่โชว์ผล
        image_url = result.get("detected_image_path") or (
            settings.MEDIA_URL + "uploads/" + os.path.basename(image.name)
        )

        # เติมแถวล่าสุดให้หน้า app
        if result.get("total") is not None:
            total = float(result["total"])
            mock_row = _mock_row(os.path.basename(image.name), total)

            latest = request.session.get("latest_batch", [])
            latest = [mock_row] + latest
            request.session["latest_batch"] = latest[:50]

            # ประวัติด้านข้าง (คงไว้เหมือนเดิม)
            history = request.session.get("recent_results", [])
            history.insert(0, {
                "filename": os.path.basename(image.name),
                "total": total,
                "date": now().strftime("%Y-%m-%d %H:%M"),
            })
            request.session["recent_results"] = history[:5]

        # ถ้าอยากเด้งไปหน้า app ให้เปิดบรรทัดด้านล่างแทน render
        # return redirect("meter_app")

    return render(request, "meter_reader/upload.html", {
        "active": "upload",
        "result": result,
        "image_url": image_url,
        "recent_results": request.session.get("recent_results", []),
        "guest": _is_guest(request) or (not request.user.is_authenticated),
    })


# ---------- Contact ----------
def contact(request):
    return render(request, "meter_reader/contact.html", {"active": "overview"})


# ---------- USB quick endpoints ----------
@never_cache
def capture_usb(request):
    cam = get_camera(index=0)
    jpeg = cam.get_jpeg()
    if jpeg is None:
        return JsonResponse({"error": "no frame"}, status=503)
    resp = HttpResponse(jpeg, content_type="image/jpeg")
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp


@never_cache
def stream(request):
    def _mjpeg_generator(cam, fps=10):
        delay = 1.0 / max(fps, 1)
        boundary = b"--frame\r\n"
        while True:
            jpeg = cam.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(delay)

    cam = get_camera(index=0)
    return StreamingHttpResponse(
        _mjpeg_generator(cam, fps=10),
        content_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


# ---------- Detail page ----------

def detection_detail(request, stem):
    # เตรียม path รูปต่าง ๆ
    summary_src = os.path.join(SUMMARY_DIR, f"{stem}_summary.jpg")
    digitdet_src = os.path.join(DETECT_DIGIT_DIR, f"{stem}_digit_detect.jpg")
    class_imgs = [
        os.path.join(CROP_CLASSX_DIR, f"{stem}_class1.png"),
        os.path.join(CROP_CLASSX_DIR, f"{stem}_class2.png"),
        os.path.join(CROP_CLASSX_DIR, f"{stem}_class3.png"),
    ]
    digit_crops_glob = os.path.join(DIGIT_CROPS_DIR, f"{stem}_*_20x32.png")

    # ------- ค่าจาก query string (ป้องกัน None และ %xx) -------
    def _q(name, default=""):
        return unquote(request.GET.get(name, default))

    digital_x = _q("digital_x")
    x01 = _q("x01")
    x001 = _q("x001")
    x0001 = _q("x0001")
    x00001 = _q("x00001")
    total = _q("total")
    digit_string = _q("digit_string") or (digital_x + x01)

    # prev/raw/final สำหรับตารางตัดสินใจ
    prev_reading = {
        "x01": _q("prev_x01") or None,
        "x001": _q("prev_x001") or None,
        "x0001": _q("prev_x0001") or None,
        "x00001": _q("prev_x00001") or None,
    }
    raw_digits = {
        "digital_x_raw": _q("raw_digital"),
        "x01_raw": _q("raw_x01"),
        "x001_raw": _q("raw_x001"),
        "x0001_raw": _q("raw_x0001"),
        "x00001_raw": _q("raw_x00001"),
    }
    final_digits = {
        "digital_x": _q("final_digital"),
        "x01": _q("final_x01"),
        "x001": _q("final_x001"),
        "x0001": _q("final_x0001"),
        "x00001": _q("final_x00001"),
    }

    # ------- รายการรูป digit crops -------
    digit_items = []
    try:
        digit_crops = sorted(glob.glob(digit_crops_glob))
        digits = list(digit_string)
        for i, p in enumerate(digit_crops):
            digit_items.append({"url": _to_media_url(p), "label": digits[i] if i < len(digits) else "?"})
    except Exception:
        digit_items = []

    # ------- รายการรูปเข็ม (analog) -------
    analog_vals = [x001, x0001, x00001]
    analog_labels = ["x001", "x0001", "x00001"]
    analog_items = []
    for i, p in enumerate(class_imgs):
        analog_items.append(
            {
                "url": _to_media_url(p),
                "label": analog_vals[i] if i < len(analog_vals) else "-",
                "cls": analog_labels[i] if i < len(analog_labels) else None,
            }
        )

    ctx = {
        "active": "upload",
        "stem": stem,
        # รูปสรุป/รูป detect digit
        "digit_summary_url": _to_media_url(summary_src),
        "digit_detect_url": _to_media_url(digitdet_src),
        # รายการรูปประกอบ
        "digit_items": digit_items,
        "analog_items": analog_items,
        # ค่าที่อ่านได้สรุป
        "digital_x": digital_x,
        "x01": x01,
        "x001": x001,
        "x0001": x0001,
        "x00001": x00001,
        "total": total,
        # ตารางตัดสินใจ
        "prev_reading": prev_reading,
        "raw_digits": raw_digits,
        "final_digits": final_digits,
    }
    return render(request, "meter_reader/detail.html", ctx)


# ===== Legacy aliases for old urls.py names =====

def overview_page(request):
    return overview(request)


def overview_view(request):
    return overview(request)


@login_or_guest
def meter_reader_app(request):
    return app(request)


@login_or_guest
def app_view(request):
    return app(request)


def usb_camera_page(request):
    # ถ้าอยากให้เปิดหน้า index เก่า
    return index(request)
