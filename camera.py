# meter_reader/camera.py
import cv2, threading, time

class USBCamera:
    def __init__(self, index=0):
        # เปิดด้วย DSHOW ก่อน ถ้าไม่ติดค่อย MSMF
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError("เปิดกล้อง USB ไม่สำเร็จ")

        # ปรับค่าที่คุณลองแล้วชัด (แก้ได้ตามต้องการ)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 50)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        self.lock = threading.Lock()
        self.frame = None
        self.running = False

    def start(self):
        if self.running: return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def get_jpeg(self, quality=85):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        if f is None:
            return None
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()

# ---- lazy singleton ----
_camera = None
def get_camera(index=0):
    global _camera
    if _camera is None:
        _camera = USBCamera(index=index)
        _camera.start()
    return _camera
