"""APT.(로제·브루노 마스) 챌린지 — 조회수 상위 9건만 추려 재검증.

analyze_apt_dance.py(11건, 조회수 무관 무작위 표본)의 후속. 이번엔 후보를 조회수로
정렬해 구조적으로 유효한(1~2인, 실시간 속도, 안무 일치) 클립 중 조회수 40만 이상만 추렸다.
조회수는 유명 크리에이터·공식 계정일 가능성을 높여 "이 영상이 실제로 그 챌린지를
수행했는가"에 대한 추가 신뢰 신호로 쓴다 — 화질·정합성 필터와 별개의 기준이다.

sample_dance_apt_hiviews/ 파일 구성과 조회수(2026-07-10 기준):
  pSzQE5dugrY  63,012,222  듀오_커플(핑크)
  GuJpK6pLDUQ  13,332,148  커튼_A(튜토리얼)
  _BYFgUDdbkY   8,847,483  듀오_커플(모방)
  ofBD5BLaBMQ   4,758,801  듀오_공원(모자)
  jGhc1WQ6dRE   1,872,230  듀오_커플(핑크)-속편   <- pSzQE5dugrY와 동일 크리에이터로 추정(같은 거실·같은 옷)
  ARW9YbBeRs8   1,367,430  듀오_필리핀(zephanie)
  7ts7UvlBwOo     711,716  듀오_아이들(twins)
  9JLhQp8pYHM     421,818  로제+켈리클락슨(talk show)  <- 안무 창시자 로제 본인 등장
  01Wp7Bp6N14     467,850  커튼_A(결과)

실행: python analyze_apt_dance_hiviews.py
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
    "pSzQE5dugrY-WE FINALLY DID THE OTHER APT. DANCE ROSÉ & Bruno M.webm": "커플_핑크_A",
    "GuJpK6pLDUQ-APT. Dance Tutorial 🪩 Viral Dance Challenge 2024.webm": "커튼_A(튜토리얼)",
    "_BYFgUDdbkY-I wanted to challenge my boyfriend and told him to.webm": "듀오_커플(모방)",
    "ofBD5BLaBMQ-APT. Dance Challenge in public! ｜ ROSÉ, Bruno Mars.webm": "듀오_공원(모자)",
    "jGhc1WQ6dRE-WHO REMEMBERS THE APT. DANCE ROSÉ & Bruno Mars! - .webm": "커플_핑크_B",
    "ARW9YbBeRs8-APT Dance Challenge with #Zephanie & #DylanMenor #.webm": "듀오_필리핀(zephanie)",
    "7ts7UvlBwOo-APT Dance Challenge Pt 2! #TWINS #rosé #brunomars .webm": "듀오_아이들(twins)",
    "9JLhQp8pYHM-Teaching #kellyclarkson the #APT dance!.webm": "로제+켈리클락슨",
    "01Wp7Bp6N14-APT. Dance Challenge Results 🪩 Viral Dance Trend 2.webm": "커튼_A(결과)",
}


def main() -> None:
    files = sorted(Path("sample_dance_apt_hiviews").glob("*"))
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
    print(f"\n쌍별 DTW 거리 — min {off_diag.min():.4f} / mean {off_diag.mean():.4f} / max {off_diag.max():.4f}")

    pairs = [(labels[i], labels[j], D[i, j]) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: p[2])
    print("\n가장 가까운 3쌍:")
    for a, b, d in pairs[:3]:
        print(f"  {a} <-> {b} : {d:.4f}")
    print("가장 먼 3쌍:")
    for a, b, d in pairs[-3:]:
        print(f"  {a} <-> {b} : {d:.4f}")

    # 동일 크리에이터 추정 쌍 2건 — 커플_핑크(A/B)와 커튼_A(튜토리얼/결과)
    for prefix in ("커플_핑크", "커튼_A"):
        same = [p for p in pairs if p[0].startswith(prefix) and p[1].startswith(prefix)]
        for a, b, d in same:
            rank = pairs.index((a, b, d)) + 1
            print(f"\n{a} <-> {b} : {d:.4f} (전체 {len(pairs)}쌍 중 {rank}번째로 가까움) — 동일 크리에이터 추정")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"APT 챌린지(조회수 상위 {n}건) — 3관절 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_apt_hiviews_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_apt_hiviews_dtw.png")


if __name__ == "__main__":
    main()
