# meter_reader/camera.py
import cv2, threading, time

class USBCamera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise RuntimeError(f"เปิดกล้อง USB ไม่สำเร็จ (index={self.index})")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 50)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        self.cap = cap

    def start(self):
        if self.running:
            return
        if self.cap is None:
            self._open()
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
        if not self.running:
            self.start()
            time.sleep(0.2)
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        if f is None:
            return None
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# ---------- singleton ใช้ร่วมทั้งแอป ----------
_camera_singleton = None
_camera_index = None

def get_camera(index=0, force_switch=False):
    """
    คืน instance เดียว ถ้า index เปลี่ยน (หรือ force_switch=True)
    จะสลับไปเปิดกล้องใหม่ให้อัตโนมัติ
    """
    global _camera_singleton, _camera_index
    if _camera_singleton is None:
        _camera_singleton = USBCamera(index=index)
        _camera_singleton.start()
        _camera_index = index
        return _camera_singleton

    if force_switch or (_camera_index != index):
        # ปิดของเดิมแล้วเปิดใหม่ด้วย index ที่ขอ
        _camera_singleton.stop()
        _camera_singleton = USBCamera(index=index)
        _camera_singleton.start()
        _camera_index = index

    return _camera_singleton
