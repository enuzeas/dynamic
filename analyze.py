"""
BIO-IP Analysis
DTW 거리 행렬 → rank-1 accuracy, EER

사용법:
  python analyze.py \
    --people "한우진:A_1.MOV,A_2.MOV,A_3.MOV" \
             "박세준:B_1.MOV,B_2.MOV,B_3.MOV"
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


def dtw(s1, s2):
    """1-D DTW, O(n·m) DP."""
    n, m = len(s1), len(s2)
    cost = np.abs(np.subtract.outer(np.asarray(s1, float), np.asarray(s2, float)))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i-1, j-1] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return D[n, m]

JOINT = "오른손목"


# ── 데이터 로딩 ─────────────────────────────────────────

def load_samples(people_args):
    """[(label, speed_array), ...] 반환."""
    samples = []
    for entry in people_args:
        name, files_str = entry.split(":", 1)
        for fp in files_str.split(","):
            fp = fp.strip()
            try:
                d = extract_dynamics(fp)
                speed = trim_motion(d[JOINT]["speed"])
                samples.append((name, speed))
                print(f"  ✓ {name}: {fp} ({len(speed)}프레임 (정렬 후))")
            except ValueError as e:
                print(f"  skip: {e}")
    return samples


# ── DTW 거리 행렬 ────────────────────────────────────────

def build_distance_matrix(samples):
    n = len(samples)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(samples[i][1], samples[j][1])
            D[i, j] = D[j, i] = d
    return D


# ── Rank-1 Accuracy ──────────────────────────────────────

def rank1_accuracy(D, labels):
    """Leave-one-out: 각 샘플의 최근접 이웃(자신 제외)이 같은 사람인지."""
    n = len(labels)
    correct = 0
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf          # 자기 자신 제외
        nn = np.argmin(row)
        if labels[nn] == labels[i]:
            correct += 1
    return correct / n


# ── EER ──────────────────────────────────────────────────

def compute_eer(D, labels):
    """
    genuine scores  : 같은 사람 쌍의 DTW 거리 (작을수록 좋음)
    impostor scores : 다른 사람 쌍의 DTW 거리
    EER             : FAR == FRR 인 임계값에서의 오류율
    """
    n = len(labels)
    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (genuine if labels[i] == labels[j] else impostor).append(D[i, j])

    genuine  = np.array(genuine)
    impostor = np.array(impostor)

    # threshold를 DTW 거리 전 범위로 스윕
    thresholds = np.linspace(
        min(genuine.min(), impostor.min()),
        max(genuine.max(), impostor.max()),
        500
    )
    far_arr, frr_arr = [], []
    for t in thresholds:
        far = (impostor <= t).mean()   # 다른 사람인데 통과 (거리가 작으면 같다고 판단)
        frr = (genuine  >  t).mean()   # 같은 사람인데 거부
        far_arr.append(far)
        frr_arr.append(frr)

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)
    idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[idx] + frr_arr[idx]) / 2

    return eer, thresholds, far_arr, frr_arr, genuine, impostor


# ── 시각화 ───────────────────────────────────────────────

def plot_results(D, labels, eer, thresholds, far_arr, frr_arr, genuine, impostor, out):
    BG, SPINE, TICK = "#0F1219", "#1E2840", "#6A7A99"
    unique = sorted(set(labels))
    person_colors = ["#2E5FC2", "#D4901E", "#2E9E5B", "#C23B8A"]

    fig = plt.figure(figsize=(16, 11), facecolor="#080A10")
    fig.suptitle("BIO-IP 다이내믹스 서명 검증", color="#D4DEEE", fontsize=15, y=0.98)
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── 1. DTW 거리 행렬 ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    im = ax.imshow(D, cmap="plasma_r", aspect="auto")
    ax.set_title("DTW 거리 행렬", color="#D4DEEE", pad=8)
    ticks = range(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=TICK)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7, color=TICK)
    # 사람별 그룹 테두리
    cumsum = 0
    for pi, name in enumerate(unique):
        sz = labels.count(name)
        rect = plt.Rectangle((cumsum - 0.5, cumsum - 0.5), sz, sz,
                              fill=False, edgecolor=person_colors[pi % 4], linewidth=2)
        ax.add_patch(rect)
        cumsum += sz
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(colors=TICK, labelsize=7)

    # ── 2. Genuine / Impostor 거리 분포 ─────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    bins = np.linspace(min(genuine.min(), impostor.min()),
                       max(genuine.max(), impostor.max()) * 1.05, 25)
    ax.hist(genuine,  bins=bins, color="#2E5FC2", alpha=0.75, label="Genuine (본인)")
    ax.hist(impostor, bins=bins, color="#D4451E", alpha=0.75, label="Impostor (타인)")
    eer_t = thresholds[np.argmin(np.abs(far_arr - frr_arr))]
    ax.axvline(eer_t, color="#F0C830", linewidth=1.5, linestyle="--", label=f"EER 임계값 {eer_t:.1f}")
    ax.set_title("거리 분포", color="#D4DEEE", pad=8)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.spines[:].set_color(SPINE)
    ax.set_xlabel("DTW 거리", color=TICK, fontsize=8)
    ax.set_ylabel("빈도", color=TICK, fontsize=8)
    ax.legend(fontsize=8, labelcolor="#D4DEEE", facecolor="#141720", edgecolor=SPINE)

    # ── 3. FAR / FRR 곡선 ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(BG)
    ax.plot(thresholds, far_arr, color="#D4451E", linewidth=2, label="FAR (타인 수락률)")
    ax.plot(thresholds, frr_arr, color="#2E5FC2", linewidth=2, label="FRR (본인 거부률)")
    ax.axvline(eer_t, color="#F0C830", linewidth=1.2, linestyle="--")
    ax.axhline(eer,   color="#F0C830", linewidth=1.2, linestyle="--", label=f"EER = {eer:.1%}")
    ax.set_title("FAR / FRR 곡선", color="#D4DEEE", pad=8)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.spines[:].set_color(SPINE)
    ax.set_xlabel("임계값 (DTW 거리)", color=TICK, fontsize=8)
    ax.set_ylabel("오류율", color=TICK, fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, labelcolor="#D4DEEE", facecolor="#141720", edgecolor=SPINE)

    # ── 4. 항상성 검증 ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(BG)
    n = len(labels)
    within_means, between_means = [], []
    for name in unique:
        idx = [i for i, l in enumerate(labels) if l == name]
        w = [D[i, j] for ii, i in enumerate(idx) for j in idx[ii+1:]]
        b = [D[i, j] for i in idx for j in range(n) if labels[j] != name]
        within_means.append(np.mean(w) if w else 0)
        between_means.append(np.mean(b) if b else 0)

    x = np.arange(len(unique))
    width = 0.35
    bars1 = ax.bar(x - width/2, within_means,  width, color="#2E5FC2", alpha=0.85, label="본인 내 (Within)")
    bars2 = ax.bar(x + width/2, between_means, width, color="#D4451E", alpha=0.85, label="타인 간 (Between)")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01, f"{h:.1f}",
                    ha="center", va="bottom", color="#D4DEEE", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(unique, fontsize=9, color=TICK)
    ax.set_title("항상성 검증\n본인↓ < 타인↑ 이면 서명 유효", color="#D4DEEE", fontsize=10, pad=8)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.spines[:].set_color(SPINE)
    ax.set_ylabel("평균 DTW 거리", color=TICK, fontsize=8)
    ax.legend(fontsize=8, labelcolor="#D4DEEE", facecolor="#141720", edgecolor=SPINE)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#080A10")
    print(f"\n저장 완료: {out}")


# ── 메인 ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", nargs="+", required=True,
                        help='이름:파일1,파일2,... 형식. 예: "한우진:A_1.MOV,A_2.MOV"')
    parser.add_argument("--out", default="analysis_output.png")
    args = parser.parse_args()

    print("영상 추출 중...")
    samples = load_samples(args.people)
    if len(samples) < 2:
        print("샘플이 2개 이상 필요합니다.")
        raise SystemExit(1)

    labels = [s[0] for s in samples]

    print(f"\nDTW 거리 행렬 계산 중 ({len(samples)}×{len(samples)})...")
    D = build_distance_matrix(samples)

    r1 = rank1_accuracy(D, labels)
    eer, thresholds, far_arr, frr_arr, genuine, impostor = compute_eer(D, labels)

    print(f"\n{'─'*40}")
    print(f"  샘플 수      : {len(samples)}개 ({len(set(labels))}명)")
    print(f"  Rank-1 Acc   : {r1:.1%}")
    print(f"  EER          : {eer:.1%}")
    print(f"{'─'*40}")

    plot_results(D, labels, eer, thresholds, far_arr, frr_arr, genuine, impostor, args.out)
