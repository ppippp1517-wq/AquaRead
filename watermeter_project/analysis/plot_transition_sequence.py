import matplotlib.pyplot as plt

# === ตัวอย่างข้อมูล (แทนค่าที่คุณดึงจากระบบจริง) ===
# ก่อน cascade (raw CNN output)
raw_digits = [4, 4, 5, 5, 6, 6, 7, 7]

# หลัง cascade (Transition + carry-over)
corrected_digits = [4, 4, 4, 5, 5, 6, 6, 7]

frames = list(range(len(raw_digits)))

# === Plot ===
plt.figure(figsize=(8, 4))
plt.plot(frames, raw_digits, 'o--', label='Raw Digit Output')
plt.plot(frames, corrected_digits, 's-', label='Corrected Digit Output')

plt.xlabel('Frame Index')
plt.ylabel('Digit Value')
plt.title('Digit Value Continuity Before and After Transition Handling')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('transition_sequence_result.png', dpi=300)
plt.show()
