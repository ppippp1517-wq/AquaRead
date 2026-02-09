import matplotlib.pyplot as plt

# รอบการหมุน (rpm)
rpm = [60, 80, 100, 120, 140]

# แรงดันที่ผลิตได้ (โวลต์) – ให้ใส่ค่าจริงที่คุณวัดแทนตัวอย่างด้านล่าง
voltage = [V60, V80, V100, V120, V140]  # เช่น [2.1, 3.0, 4.2, 5.1, 6.0]

plt.figure(figsize=(8, 5))
plt.plot(rpm, voltage, marker='o')
plt.title("กราฟแสดงความสัมพันธ์ระหว่างความเร็วรอบกับแรงดันที่ผลิตได้")
plt.xlabel("ความเร็วรอบ (รอบ/นาที)")
plt.ylabel("แรงดันไฟฟ้า (โวลต์)")
plt.grid(True)
plt.tight_layout()
plt.show()