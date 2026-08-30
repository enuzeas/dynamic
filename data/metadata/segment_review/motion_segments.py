import csv
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def mmss(x, _pos=None):
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"

proxy_path, fps = sys.argv[1], float(sys.argv[2])
min_pause_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
out_prefix = sys.argv[4] if len(sys.argv) > 4 else "out"

cap = cv2.VideoCapture(proxy_path)
prev = None
energy = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = frame[:, :, 0].astype(np.float32)  # already grayscale, single channel is enough
    if prev is not None:
        energy.append(np.abs(frame - prev).mean())
    prev = frame
cap.release()
energy = np.array(energy)

# smooth with a small moving average to kill single-frame noise
k = max(1, int(fps))  # ~1 second window
kernel = np.ones(k) / k
smooth = np.convolve(energy, kernel, mode="same")

threshold = np.percentile(smooth, 25)
is_low = smooth < threshold

# find contiguous low-motion runs of at least ~0.8s -> candidate pauses
min_run = int(fps * min_pause_s)
runs = []
start = None
for i, low in enumerate(is_low):
    if low and start is None:
        start = i
    elif not low and start is not None:
        if i - start >= min_run:
            runs.append((start, i))
        start = None
if start is not None and len(is_low) - start >= min_run:
    runs.append((start, len(is_low)))


def t(frame_idx):
    return frame_idx / fps


print(f"total duration ~{len(energy)/fps:.1f}s, {len(energy)} frames, threshold={threshold:.2f}")
print(f"found {len(runs)} pause candidates:")
for s, e in runs:
    print(f"  pause {t(s):7.1f}s - {t(e):7.1f}s  (dur {t(e)-t(s):.1f}s)")

# derive "active" segments = gaps between pauses
segments = []
prev_b = 0
for s, e in runs:
    if s > prev_b:
        segments.append((prev_b, s))
    prev_b = e
if prev_b < len(energy):
    segments.append((prev_b, len(energy)))

print("\nactive segments (candidate actions):")
for i, (s, e) in enumerate(segments, 1):
    print(f"  seg {i}: {t(s):7.1f}s - {t(e):7.1f}s  (dur {t(e)-t(s):.1f}s)")

csv_path = f"{out_prefix}_segments.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seg", "start_s", "end_s", "duration_s"])
    for i, (s, e) in enumerate(segments, 1):
        w.writerow([i, f"{t(s):.1f}", f"{t(e):.1f}", f"{t(e)-t(s):.1f}"])
print(f"\ncsv written: {csv_path}")

png_path = f"{out_prefix}_energy.png"
times = np.arange(len(smooth)) / fps
plt.figure(figsize=(18, 4))
plt.plot(times, smooth, linewidth=0.8)
plt.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, label="pause threshold")
for s, e in runs:
    plt.axvspan(t(s), t(e), color="red", alpha=0.15)
for i, (s, e) in enumerate(segments, 1):
    mid = (t(s) + t(e)) / 2
    plt.text(mid, smooth.max() * 0.95, str(i), ha="center", fontsize=8)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(mmss))
ax.xaxis.set_major_locator(MultipleLocator(15))
plt.xlim(0, times[-1])
plt.xlabel("time (mm:ss)")
plt.ylabel("motion energy")
plt.title("motion energy over time (red = candidate pause, numbers = candidate action segments)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(png_path, dpi=120)
print(f"plot written: {png_path}")
