import easyocr

def process_ocr(cropped_image):
    # ใช้ EasyOCR อ่านตัวเลขจากภาพที่ตัด
    reader = easyocr.Reader(['en'])  # ใช้ภาษาอังกฤษ
    result = reader.readtext(cropped_image)
    return result
