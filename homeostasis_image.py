"""항상성 검증: 동작 시작점 정렬 후 ICC + 반복 상관 실측"""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from demo import extract_dynamics, trim_motion

JOINT = "오른손목"
THRESHOLD = 0.08

PEOPLE = [
    ("이한성", ["sample/compressed/A_1.mp4", "sample/compressed/A_2.mp4", "sample/compressed/A_3.mp4"]),
    ("송필순", ["sample/compressed/B_1.mp4", "sample/compressed/B_2.mp4", "sample/compressed/B_3.mp4"]),
    ("박세준", ["sample/compressed/C_1.mp4", "sample/compressed/C_2.mp4", "sample/compressed/C_3.mp4"]),
]


def pearsonr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def icc_1_1(trials):
    """ICC(1,1) — treats time points as subjects, trials as raters."""
    data = np.array(trials)   # (k, n)
    k, n = data.shape
    grand = data.mean()
    row_means = data.mean(axis=0)
    ms_b = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_w = np.sum((data - row_means) ** 2) / (n * (k - 1))
    return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w)) if ms_b > ms_w else 0.0


def load_and_align(files):
    speeds = []
    for f in files:
        s = trim_motion(extract_dynamics(f)[JOINT]["speed"], THRESHOLD)
        speeds.append(s)
    min_len = min(len(s) for s in speeds)
    return [s[:min_len] for s in speeds]


def compute_metrics(files):
    speeds = load_and_align(files)
    k = len(speeds)
    rs = [pearsonr(speeds[i], speeds[j]) for i in range(k) for j in range(i+1, k)]
    return icc_1_1(speeds), float(np.mean(rs))


# ── 계산 ─────────────────────────────────────────────────
results = {}
for name, files in PEOPLE:
    print(f"  처리 중: {name}...")
    icc, mean_r = compute_metrics(files)
    results[name] = (icc, mean_r)
    print(f"    ICC={icc:.4f}  mean_r={mean_r:.4f}")

names     = list(results.keys())
icc_vals  = [results[n][0] for n in names]
corr_vals = [results[n][1] for n in names]

# ── 시각화 ───────────────────────────────────────────────
bg_color  = [0.03, 0.04, 0.06]
axis_bg   = [0.07, 0.08, 0.11]
text_color= [0.84, 0.88, 0.93]
tick_color= [0.55, 0.62, 0.72]
colors    = [[0.18, 0.37, 0.76], [0.83, 0.56, 0.12], [0.18, 0.62, 0.35]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.2), facecolor=bg_color)
x = np.arange(len(names))
width = 0.55

for ax, vals, ylabel, title in [
    (ax1, icc_vals,  "ICC",         "Homeostasis: ICC"),
    (ax2, corr_vals, "correlation",  "Repeat Similarity: mean r"),
]:
    ax.bar(x, vals, width, color=colors)
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color=[0.8, 0.8, 0.8], linestyle="--", linewidth=1.5, label="weak threshold")
    ax.set_title(title, color=text_color, fontsize=12)
    ax.set_ylabel(ylabel, color=tick_color)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_facecolor(axis_bg)
    ax.tick_params(colors=tick_color)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(tick_color)
    ax.grid(True, alpha=0.3, color=tick_color)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", color=text_color, fontsize=9)

fig.suptitle("Low Homeostasis in Wrist-Speed Repeats (onset-aligned)",
             color=text_color, fontsize=13)
plt.tight_layout()
plt.savefig("homeostasis_matlab.png", dpi=180, bbox_inches="tight", facecolor=bg_color)
print("✓ homeostasis_matlab.png 생성 완료")
