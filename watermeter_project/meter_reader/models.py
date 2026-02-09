from django.db import models

class Meter(models.Model):
    meter_id = models.CharField(max_length=64, unique=True)  # เช่น SPP-01, บ้านเลขที่, ฯลฯ
    name     = models.CharField(max_length=128, blank=True, default="")

    def __str__(self):
        return self.meter_id

class MeterReading(models.Model):
    meter      = models.ForeignKey(Meter, on_delete=models.CASCADE, related_name="readings")
    timestamp  = models.DateTimeField(auto_now_add=True)

    # ส่วนค่าที่อ่านได้ทีละหลัก
    digital_x  = models.CharField(max_length=10)  # จำนวนเต็มฝั่งซ้าย (string ตามที่คุณเก็บ)
    x01        = models.IntegerField()            # ทศนิยมตำแหน่งที่ 1
    x001       = models.IntegerField()            # ทศนิยมตำแหน่งที่ 2
    x0001      = models.IntegerField()            # ทศนิยมตำแหน่งที่ 3
    x00001     = models.IntegerField()            # ทศนิยมตำแหน่งที่ 4
    total      = models.CharField(max_length=32)  # “0003.1590” เก็บเป็นสตริงเพื่อคงรูป

    detected_image_path  = models.CharField(max_length=255, blank=True, default="")
    digit_detect_path    = models.CharField(max_length=255, blank=True, default="")
    digit_summary_path   = models.CharField(max_length=255, blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["meter", "-timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.meter.meter_id} @ {self.timestamp:%Y-%m-%d %H:%M:%S} = {self.total}"
