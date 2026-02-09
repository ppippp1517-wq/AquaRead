from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.overview_page, name="overview"),
    path("start/", views.start_choice, name="start"),
    path("start/guest/", views.start_guest, name="start_guest"),

    path("login/",  auth_views.LoginView.as_view(
        template_name="meter_reader/login.html",
        redirect_authenticated_user=True
    ), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),

    # จุดเข้าแอพใหม่
    path("enter/", views.app_entry, name="app_entry"),

    # หน้าใช้งานจริง
    path("app/", views.meter_reader_app, name="meter_app"),

    # อื่น ๆ
    path('webapp/', views.index, name='index'),
    path('api/analyze/', views.analyze, name='analyze'),
    path('upload/', views.upload_image, name='upload_image'),
    path('upload_folder/', views.upload_folder, name='upload_folder'),
    path('capture_image/', views.capture_image, name='capture_image'),
    path('preview_image/', views.preview_image, name='preview_image'),
    path('confirm_image/', views.confirm_image, name='confirm_image'),
    path('contact/', views.contact, name='contact'),
    path('capture/', views.capture_usb, name='capture_usb'),
    path('camera/', views.usb_camera_page, name='usb_camera_page'),
    path('stream/', views.stream, name='stream'),
    path("detail/<slug:stem>/", views.detection_detail, name="detection_detail"),
]
