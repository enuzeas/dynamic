"""'Take Me Up'(Ralphi Rosario, 1997곡 2026년 리믹스로 재유행) 챌린지 — n=2, 참고용.

사용자가 유튜브 쇼츠 하나(tFePzSW3AVE)를 보내며 이 챌린지로 분석을 요청했다.
검색(WebSearch 다수 쿼리 + yt-dlp ytsearch 직접 조회)을 조회수 기준 없이까지
낮춰 폭넓게 시도했지만, 이 챌린지의 실사·1~2인·YouTube 쇼츠 커버는 단 2개만
확인됐다 — 나머지 후보는 전부 부적합(대형 댄스 스튜디오 클래스 10인 이상,
6분 넘는 셔플댄스 강습, 다른 챌린지의 오검색 등)했다.

포켓댄스(doc/golden_dance_analysis.md 0번 참고)와 같은 패턴 — 챌린지 자체는
실재하지만(TikTok 쪽에 진짜 바이럴이 몰려 있는 것으로 보임) YouTube 전용
표본은 얇다. n=2, 1쌍이라 DTW 거리 행렬로서의 통계적 의미는 없고,
"두 실사 커버가 서로 얼마나 다른가" 참고 수치만 낸다.

최종 2건 (2026-07-10 기준 조회수):
  tFePzSW3AVE   2,329,776  솔로_발코니(사용자 제공)
  dj9IT2AC3iY   3,575,691  솔로_주방(에이프런 챌린지)

실행: python analyze_takemeup_dance.py
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
    "tFePzSW3AVE-story-ai-real.webm": "솔로_발코니(233만)",
    "dj9IT2AC3iY-kitchen-apron.webm": "솔로_주방(358만)",
}


def main() -> None:
    files = sorted(Path("sample_dance_takemeup").glob("*"))
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

    a, b = labels[0], labels[1]
    print(f"\n{a} <-> {b} : {D[0, 1]:.4f} (경로 길이 정규화 DTW, n=2라 분포 비교는 불가 — 단일 참고값)")

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title("Take Me Up 챌린지(n=2, 참고용) — 3관절 평균 DTW 거리", color="#D4DEEE", pad=10, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_takemeup_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_takemeup_dtw.png")


if __name__ == "__main__":
    main()
