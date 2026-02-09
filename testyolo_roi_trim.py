import cv2
from yolo_roi_trim import trim_to_digit_window

img = cv2.imread(r"D:\projectCPE\class1.png")
trimmed = trim_to_digit_window(img)
cv2.imwrite(r"D:\projectCPE\class1_trimmed.png", trimmed)
