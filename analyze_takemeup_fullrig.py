"""'Take Me Up' 실사 vs AI의심 — 3관절(오른팔만) 대신 8관절(양팔+양엉덩이) 전체 리그로 재검증.

analyze_takemeup_ai_vs_real.py는 오른손목·팔꿈치·어깨 3관절만 썼다. "외형을 지우고
움직임 리그만 남기면 결과가 달라지는가"라는 질문에 답하기 위해, pose_extract.py가
지원하는 8관절(양쪽 손목·팔꿈치·어깨·엉덩이 — scope.md MVP 스펙과 동일) 전체로
같은 비교를 반복한다. MediaPipe 추출 자체가 이미 외형(얼굴·옷·배경)을 버리고
관절 좌표만 남기므로, 이번 비교는 "외형 제거 여부"가 아니라 "얼마나 많은 관절을
보는가"의 효과를 테스트하는 것이다.

실행: python analyze_takemeup_fullrig.py
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
from pose_extract import extract_joint_dynamics, LANDMARK_INDEX
from analyze_takemeup_ai_vs_real import LABELS

JOINTS = tuple(LANDMARK_INDEX)  # 8관절: 양쪽 손목·팔꿈치·어깨·엉덩이

TEMPLATE7 = [
    "AI_주방_중국1", "AI_에펠탑_에이미", "AI_주방_중국2", "AI_산전망_자칭AI",
    "AI_에펠탑_두진위2", "AI_리조트_두진위3", "AI_스튜디오_두진위1",
]


def main() -> None:
    files = sorted(Path("sample_dance_takemeup_full").glob("*"))
    speeds: dict[str, dict[str, np.ndarray]] = {}
    for f in files:
        label = LABELS.get(f.name, f.stem)
        dyn = extract_joint_dynamics(str(f), joints=JOINTS)
        speeds[label] = {j: trim_motion(dyn[j]["speed"]) for j in JOINTS}
        print(f"추출 완료: {label} ({len(JOINTS)}관절)")

    labels = list(speeds)
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            per_joint = [dtw(speeds[labels[i]][jt], speeds[labels[j]][jt]) for jt in JOINTS]
            D[i, j] = D[j, i] = float(np.mean(per_joint))

    real_idx = [i for i, l in enumerate(labels) if l.startswith("R_")]
    ai_idx = [i for i, l in enumerate(labels) if l.startswith("AI_")]
    t_idx = [labels.index(l) for l in TEMPLATE7]
    o_idx = [i for i in range(n) if labels[i] not in TEMPLATE7]

    within_real = np.array([D[i, j] for a, i in enumerate(real_idx) for j in real_idx[a + 1:]])
    within_ai = np.array([D[i, j] for a, i in enumerate(ai_idx) for j in ai_idx[a + 1:]])
    within_t = np.array([D[i, j] for a, i in enumerate(t_idx) for j in t_idx[a + 1:]])
    within_o = np.array([D[i, j] for a, i in enumerate(o_idx) for j in o_idx[a + 1:]])

    print(f"\n[8관절 전체 리그 결과]")
    print(f"실사 8건 내부 — mean {within_real.mean():.4f} / max {within_real.max():.4f}")
    print(f"AI의심 9건 내부 — mean {within_ai.mean():.4f} / max {within_ai.max():.4f}")
    print(f"템플릿7 내부 — mean {within_t.mean():.4f} / max {within_t.max():.4f} (n={len(within_t)}쌍)")
    print(f"나머지10 내부 — mean {within_o.mean():.4f} / max {within_o.max():.4f} (n={len(within_o)}쌍)")

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7, color="#D4DEEE")
    ax.set_title("Take Me Up — 8관절 전체 리그 DTW 거리 (실사 vs AI의심)", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_takemeup_fullrig_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_takemeup_fullrig_dtw.png")


if __name__ == "__main__":
    main()
