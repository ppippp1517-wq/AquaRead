import cv2
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load YOLO model (detects analog meter)
model = YOLO('best.pt')
names = model.names

# Setup Gemini AI
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# Open video
cap = cv2.VideoCapture("meter.mp4")
frame_count = 0

# Timing
last_sent_time = 0
send_interval = 5  # seconds

# Folder for today
today_str = datetime.now().strftime("%Y-%m-%d")
crop_folder = f"crop_{today_str}"
os.makedirs(crop_folder, exist_ok=True)

# Global to hold latest Gemini reading
latest_reading = "Reading: ..."

# Gemini processing in background thread
def process_crop_async(image_path):
    global latest_reading
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Read the analog meter and return only the numeric reading no need any description(e.g., 01425)"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = gemini_model.invoke([message])
        reading = response.content.strip()
        latest_reading = f"Reading: {reading}"  # Update global display value
        print(f"{os.path.basename(image_path)} - {latest_reading}")

        # Save reading as .txt file next to image
        txt_path = image_path.replace(".jpg", ".txt")
        with open(txt_path, "w") as f:
            f.write(latest_reading)
    except Exception as e:
        print("Gemini Error:", e)

# Mouse position debug
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Main loop
crop_preview = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            label = names[class_ids[i]]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display reading on top-left of bounding box
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 200, y1), (0, 0, 0), -1)
            cv2.putText(frame, latest_reading, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Crop & send to Gemini every 5 sec
            current_time = time.time()
            if current_time - last_sent_time >= send_interval:
                last_sent_time = current_time

                crop_img = frame[y1:y2, x1:x2]
                crop_preview = crop_img.copy()
                timestamp = int(current_time)
                crop_filename = os.path.join(crop_folder, f"crop_{timestamp}.jpg")
                cv2.imwrite(crop_filename, crop_img)

                threading.Thread(target=process_crop_async, args=(crop_filename,), daemon=True).start()
                break  # process only 1 crop per frame

     # Show mini crop preview in bottom-right corner
    if crop_preview is not None:
        preview = cv2.resize(crop_preview, (200, 150))
        h, w, _ = preview.shape
        frame[-h:, -w:] = preview  # embed preview in bottom-right
        cv2.putText(frame, "CROP", (frame.shape[1] - w, frame.shape[0] - h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)



    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
