from alignment_folder import align_folder_images

# กำหนด path (ต้องตรงกับโครงสร้างโฟลเดอร์)
reference_img = "Reference.jpg"       # ชื่อไฟล์ reference (ระวังตัวพิมพ์ใหญ่/เล็ก)
input_folder = "test_images"          # โฟลเดอร์ภาพที่จะจัดตำแหน่ง
output_folder = "output_aligned"      # โฟลเดอร์บันทึกผลลัพธ์

# เรียกฟังก์ชัน Alignment
results = align_folder_images(reference_img, input_folder, output_folder)

# แสดงสรุปผล
print("\nSummary of Alignment:")
for r in results:
    print(f"{r['filename']} - Matches: {r['matches']} - Quality: {r['quality']}")
