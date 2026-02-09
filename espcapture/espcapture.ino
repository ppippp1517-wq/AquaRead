#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "FS.h"
#include "SPIFFS.h"

// กำหนดค่า GPIO ของกล้อง ESP32
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// กำหนดพินไฟแฟลช (GPIO 4)
#define FLASH_LED_PIN 4  // กำหนดพินไฟแฟลช

// กำหนดค่า Wi-Fi
const char* ssid = "Comen1301";  // Your WiFi SSID
const char* password = "comen1301"; 


// ตั้งค่าเซิร์ฟเวอร์ HTTP
httpd_handle_t camera_httpd = NULL;

// ฟังก์ชันที่ใช้รับคำขอถ่ายภาพ
esp_err_t capture_handler(httpd_req_t *req) {
    Serial.println("Received capture request!");  // พิมพ์ข้อความเพื่อดีบัก
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // เพิ่ม CORS header เพื่ออนุญาตการเชื่อมต่อจาก localhost หรือจากแหล่งที่มาที่ต้องการ
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");  // อนุญาตทุกแหล่งที่มา

    // ส่งภาพกลับไปยังไคลเอนต์
    httpd_resp_set_type(req, "image/jpeg");
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);

    

    esp_camera_fb_return(fb);

    return res;
}

// ฟังก์ชันเริ่มต้นเซิร์ฟเวอร์ HTTP
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  // ลงทะเบียนฟังก์ชัน capture_handler กับ URL "/capture"
  httpd_uri_t capture_uri = {
    .uri       = "/capture",
    .method    = HTTP_GET,
    .handler   = capture_handler,
    .user_ctx  = NULL
  };

  // เริ่มต้นเซิร์ฟเวอร์ HTTP
  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    Serial.println("Camera server started.");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_LED_PIN, OUTPUT);  // กำหนดพินสำหรับไฟแฟลช

  // เริ่มต้น SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  // เชื่อมต่อ Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  
  int connectionTimeout = 30;  // Timeout 30 seconds
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    connectionTimeout--;
    if (connectionTimeout == 0) {
      Serial.println();
      Serial.println("Failed to connect to Wi-Fi. Restarting ESP32.");
      ESP.restart();  // Restart the ESP32 if unable to connect
    }
  }

  // เชื่อมต่อสำเร็จ
  Serial.println();
  Serial.println("Successfully connected to WiFi!");
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());  // แสดง IP ที่ได้รับจาก Wi-Fi

  // เริ่มต้นกล้อง
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 8;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }

  // เริ่มต้นเซิร์ฟเวอร์ HTTP
  startCameraServer();
}

void loop() {
  // ไม่มีการทำงานใน loop, เซิร์ฟเวอร์ HTTP จะจัดการคำขอทั้งหมด
}
