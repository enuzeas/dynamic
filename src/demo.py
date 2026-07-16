"""
BIO-IP Quick Demo
같은 동작을 한 두 사람의 다이내믹스 서명을 나란히 시각화한다.

사용법 (repo 루트에서):
  python src/demo.py --a sample/original/A_1.mp4 sample/original/A_2.mp4 sample/original/A_3.mp4 \
                     --b sample/original/B_1.mp4 sample/original/B_2.mp4 sample/original/B_3.mp4 \
                     --names "한우진" "박세준"
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pose_extract import extract_joint_dynamics as extract_dynamics


def normalize_length(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """길이가 다른 시계열을 가장 짧은 것에 맞춰 잘라낸다."""
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]


def trim_motion(speed: np.ndarray, threshold: float = 0.15, pad: int = 3) -> np.ndarray:
    """움직임 구간만 남긴다. threshold는 절대값이 아니라 이 시계열 최고 속도 대비 비율(0~1) —
    MediaPipe 정규화 좌표 기반 속도는 영상마다 스케일이 다르므로 절대 threshold는 못 쓴다."""
    moving = np.flatnonzero(speed > threshold * speed.max())
    if len(moving) == 0:
        return speed
    start = max(0, moving[0] - pad)
    end = min(len(speed), moving[-1] + pad + 1)
    return speed[start:end]


def plot_comparison(files_list, names, out="demo_output.png", threshold=0.15):
    joint = "오른손목"
    palette = [
        ["#2E5FC2", "#4A7FE0", "#6A9FF8"],
        ["#D4901E", "#E8B050", "#F0C878"],
        ["#2E9E5B", "#48C278", "#78DFA0"],
    ]
    avg_colors = ["#2E5FC2", "#D4901E", "#2E9E5B"]

    n = len(files_list)
    fig = plt.figure(figsize=(6 * n, 10), facecolor="#080A10")
    fig.suptitle(
        f"다이내믹스 서명 비교 — {joint} 속도\n같은 동작, 다른 사람",
        color="#D4DEEE", fontsize=14, y=0.98
    )

    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    ax_avg = fig.add_subplot(gs[1, :])

    all_speeds = []
    for pi, (files, ax) in enumerate(zip(files_list, axes)):
        speeds = []
        for i, fp in enumerate(files):
            try:
                d = extract_dynamics(fp)
                s = trim_motion(d[joint]["speed"], threshold)
                speeds.append(s)
                ax.plot(s, color=palette[pi][i % 3], alpha=0.8, linewidth=1.2, label=f"반복 {i+1}")
            except ValueError as e:
                print(f"  skip: {e}")
        all_speeds.append(speeds)

        ax.set_facecolor("#0F1219")
        ax.set_title(names[pi], color=avg_colors[pi], fontsize=12, pad=8)
        ax.tick_params(colors="#6A7A99", labelsize=8)
        ax.spines[:].set_color("#1E2840")
        ax.set_xlabel("프레임", color="#6A7A99", fontsize=8)
        ax.set_ylabel("속도 (정규화)", color="#6A7A99", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#6A7A99", facecolor="#141720", edgecolor="#1E2840")

    ax_avg.set_facecolor("#0F1219")
    ax_avg.set_title(f"평균 속도 곡선 비교 — {n}명의 서명이 다른가?",
                     color="#D4DEEE", fontsize=11, pad=8)

    for pi, speeds in enumerate(all_speeds):
        if speeds:
            normed = normalize_length(speeds)
            mean = np.mean(normed, axis=0)
            std  = np.std(normed, axis=0)
            x = np.arange(len(mean))
            ax_avg.plot(x, mean, color=avg_colors[pi], linewidth=2, label=names[pi])
            ax_avg.fill_between(x, mean - std, mean + std, color=avg_colors[pi], alpha=0.15)

    ax_avg.tick_params(colors="#6A7A99", labelsize=8)
    ax_avg.spines[:].set_color("#1E2840")
    ax_avg.set_xlabel("프레임 (정규화)", color="#6A7A99", fontsize=8)
    ax_avg.set_ylabel("속도", color="#6A7A99", fontsize=8)
    ax_avg.legend(fontsize=10, labelcolor="#D4DEEE", facecolor="#141720", edgecolor="#1E2840")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#080A10")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    parser.add_argument("--c", nargs="+", default=None)
    parser.add_argument("--names", nargs="+", default=["Person A", "Person B", "Person C"])
    parser.add_argument("--out", default="demo_output.png")
    parser.add_argument("--threshold", type=float, default=0.15, help="최고 속도 대비 비율 (0~1)")
    args = parser.parse_args()

    files_list = [args.a, args.b]
    if args.c:
        files_list.append(args.c)
    for i, files in enumerate(files_list):
        print(f"추출 중: {args.names[i]} ({len(files)}개 영상)...")
    plot_comparison(files_list, names=args.names, out=args.out, threshold=args.threshold)
