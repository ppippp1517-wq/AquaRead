# watermeter_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", include("meter_reader.urls")),                  # เส้นทางแอป
    path("accounts/", include("django.contrib.auth.urls")),  # ✅ เส้นทาง login/logout ฯลฯ
    path("admin/", admin.site.urls),
]

# เสิร์ฟสื่อใน dev
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)