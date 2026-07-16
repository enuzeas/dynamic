"""'Take Me Up' 챌린지 — 실사로 확인된 클립끼리만("다른 사람" 표본).

analyze_takemeup_ai_vs_real.py에서 실사(R_)로 분류한 8건만 골라 킥드럼·애플·APT와
같은 방식(3관절 평균 DTW, 경로 길이 정규화)으로 "같은 챌린지, 다른 사람" 거리 분포를 낸다.

실행: python analyze_takemeup_real.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from analyze import dtw
from demo import trim_motion
from pose_extract import extract_joint_dynamics

JOINTS = ("오른손목", "오른팔꿈치", "오른어깨")

LABELS = {
    "REAL_gV3yefAb1Ao.webm": "공원_서아",
    "REAL_BUZAGZG2ZsQ.webm": "밀밭_중국",
    "REAL_q-6X-Ektd9Q.webm": "공원_은빈",
    "REAL_CbbRIn1ibm4.webm": "행사장_나율",
    "REAL_MHkk9onWMuo.webm": "골목_중국",
    "REAL_vmoZIJg-X20.webm": "주차장_수빈(듀오)",
    "REAL_dj9IT2AC3iY.webm": "주방_에이프런",
    "REAL_WyPYv01TFmg.webm": "스튜디오_체리",
}


def main() -> None:
    files = [p for p in sorted(Path("sample_dance_takemeup_full").glob("*")) if p.name.startswith("REAL_")]
    speeds: dict[str, dict[str, np.ndarray]] = {}
    for f in files:
        label = LABELS.get(f.name, f.stem)
        print(f"추출 중: {label} ({f.name})")
        dyn = extract_joint_dynamics(str(f), joints=JOINTS)
        speeds[label] = {j: trim_motion(dyn[j]["speed"]) for j in JOINTS}
        lengths = ", ".join(f"{j}:{len(speeds[label][j])}" for j in JOINTS)
        print(f"  -> {lengths} (움직임 구간, 관절별)")

    labels = list(speeds)
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            per_joint = [dtw(speeds[labels[i]][jt], speeds[labels[j]][jt]) for jt in JOINTS]
            D[i, j] = D[j, i] = float(np.mean(per_joint))

    off_diag = D[np.triu_indices(n, k=1)]
    print(f"\n쌍별 DTW 거리(경로 길이 정규화) — min {off_diag.min():.4f} / mean {off_diag.mean():.4f} / max {off_diag.max():.4f}")

    pairs = [(labels[i], labels[j], D[i, j]) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: p[2])
    print("\n가장 가까운 3쌍:")
    for a, b, d in pairs[:3]:
        print(f"  {a} <-> {b} : {d:.4f}")
    print("가장 먼 3쌍:")
    for a, b, d in pairs[-3:]:
        print(f"  {a} <-> {b} : {d:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"Take Me Up 챌린지(실사 {n}건) — 3관절 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_takemeup_real_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_takemeup_real_dtw.png")


if __name__ == "__main__":
    main()
