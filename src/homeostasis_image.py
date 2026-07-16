"""항상성 검증: percentage-of-duration 정렬 후 ICC + 반복 상관 실측

사용법:
  python homeostasis_image.py \
    --people "이한성:sample/compressed/A_1.mp4,sample/compressed/A_2.mp4,sample/compressed/A_3.mp4" \
             "송필순:sample/compressed/B_1.mp4,sample/compressed/B_2.mp4,sample/compressed/B_3.mp4"
"""
import argparse
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
THRESHOLD = 0.15  # 최고 속도 대비 비율 — demo.trim_motion 참고

DEFAULT_PEOPLE = [
    "이한성:sample/compressed/A_1.mp4,sample/compressed/A_2.mp4,sample/compressed/A_3.mp4",
    "송필순:sample/compressed/B_1.mp4,sample/compressed/B_2.mp4,sample/compressed/B_3.mp4",
    "박세준:sample/compressed/C_1.mp4,sample/compressed/C_2.mp4,sample/compressed/C_3.mp4",
]


def parse_people(people_args: list[str]) -> list[tuple[str, list[str]]]:
    """analyze.py --people과 동일한 '이름:파일1,파일2,...' 형식."""
    parsed = []
    for entry in people_args:
        name, files_str = entry.split(":", 1)
        parsed.append((name, [f.strip() for f in files_str.split(",")]))
    return parsed


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


def resample_fraction(speed: np.ndarray, n: int = 50) -> np.ndarray:
    """움직임 구간을 0~100% 진행률로 놓고 고정 길이 n으로 리샘플링한다.

    DTW로 억지로 맞추는 방법을 먼저 시도했으나 폐기했다 — DTW는 두 곡선을
    최대한 비슷하게 만드는 게 목적이라, 타인 간 데이터끼리도 정렬 후 상관이
    0.85+로 튀었다(실측 확인됨). 그런데 이 프로젝트의 전제 자체가 "타이밍
    차이가 개인 서명"이므로, 타이밍을 지워버리는 정렬 방법으로 항상성을
    재는 건 자기모순이다. 시간을 진행률(%)로만 맞추면 절대 프레임 수 차이
    (촬영 속도차)는 흡수하면서 국소적 타이밍 차이는 보존된다.
    """
    x_old = np.linspace(0, 1, len(speed))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, speed)


def load_and_align(files, n_points: int = 50):
    speeds = [trim_motion(extract_dynamics(f)[JOINT]["speed"], THRESHOLD) for f in files]
    return [resample_fraction(s, n_points) for s in speeds]


def within_subject_stats(speeds):
    k = len(speeds)
    rs = [pearsonr(speeds[i], speeds[j]) for i in range(k) for j in range(i + 1, k)]
    return icc_1_1(speeds), float(np.mean(rs))


def between_subject_r(speeds_by_person: dict) -> float:
    """타인 간 평균 상관 — 본인 내 상관(within)이 이 값보다 뚜렷이 높아야
    항상성 신호가 진짜라고 볼 수 있다. 안 그러면 정렬 방법이 모두를 비슷하게
    만드는 것일 뿐일 수 있다(DTW 시도에서 실제로 벌어진 문제)."""
    names = list(speeds_by_person)
    rs = [
        pearsonr(a, b)
        for i in range(len(names))
        for j in range(i + 1, len(names))
        for a in speeds_by_person[names[i]]
        for b in speeds_by_person[names[j]]
    ]
    return float(np.mean(rs)) if rs else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", nargs="+", default=DEFAULT_PEOPLE,
                        help='이름:파일1,파일2,... 형식. 예: "한우진:A_1.mp4,A_2.mp4"')
    parser.add_argument("--out", default="homeostasis_matlab.png")
    args = parser.parse_args()
    PEOPLE = parse_people(args.people)

    # ── 계산 ─────────────────────────────────────────────
    results = {}
    speeds_by_person = {}
    for name, files in PEOPLE:
        print(f"  처리 중: {name}...")
        speeds = load_and_align(files)
        speeds_by_person[name] = speeds
        icc, mean_r = within_subject_stats(speeds)
        results[name] = (icc, mean_r)
        print(f"    ICC={icc:.4f}  mean_r={mean_r:.4f}")

    between_r = between_subject_r(speeds_by_person)
    print(f"\n  [대조군] 타인 간 평균 상관: mean_r={between_r:.4f}")
    print("  → 본인 내 상관이 이보다 뚜렷이 높아야 항상성 신호가 유효하다고 볼 수 있다\n")

    names     = list(results.keys())
    icc_vals  = [results[n][0] for n in names]
    corr_vals = [results[n][1] for n in names]

    # ── 시각화 ───────────────────────────────────────────
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

    fig.suptitle("Homeostasis in Wrist-Speed Repeats (%-of-duration aligned)",
                 color=text_color, fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight", facecolor=bg_color)
    print(f"✓ {args.out} 생성 완료")
