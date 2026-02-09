# plot_transition_fig12.py
import matplotlib.pyplot as plt

# ===============================
# INPUT DATA (แก้ตรงนี้ได้)
# ===============================

# Frame index
frames = list(range(8))

# Ground truth digit values (ค่าจริงตามมิเตอร์)
ground_truth = [4, 4, 5, 5, 6, 6, 7, 7]

# Raw CNN prediction (ก่อน transition handling)
raw_prediction = [4, 4, 5, 5, 6, 6, 7, 7]

# ตัวอย่างเคส error (ถ้าจะโชว์ rollover ชัด ๆ)
# raw_prediction = [4, 4, 5, 4, 6, 5, 7, 6]

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(8, 4))

plt.plot(
    frames,
    ground_truth,
    marker="o",
    linewidth=2,
    label="Ground Truth",
)

plt.plot(
    frames,
    raw_prediction,
    marker="x",
    linestyle="--",
    linewidth=2,
    label="Raw CNN Output (No Transition Handling)",
)

plt.xlabel("Frame Index")
plt.ylabel("Digit Value")
plt.title("Digit Prediction Before Transition Handling")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
