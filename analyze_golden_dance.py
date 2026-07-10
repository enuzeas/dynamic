"""'Golden'(HUNTR/X, KPop Demon Hunters) 챌린지 — 조회수 상위 4건, 포켓댄스 대체 파일럿.

포켓댄스(PokéDance)를 조회수 기준으로 추리려 했더니 고조회수 유튜브 콘텐츠 대부분이
2D 마스코트 애니메이션·3D 아바타(Zepeto)·VTuber 편집물이라 실제 사람 관절을 추출할
대상 자체가 없었다(실사 커버는 TikTok 쪽에 몰려 있는 것으로 보임). 그래서 실사 커버가
풍부한 다른 초대형 챌린지로 교체했다 — 'Golden'은 넷플릭스 케이팝 데몬 헌터스 OST로,
MV 자체가 유튜브 10억 뷰를 넘긴 2025년 최대 히트곡 중 하나이며 실사 댄스 커버가 매우 많다.

후보 9개 중 3인 초과 그룹(대형 댄스 학원 클래스)·리액션/노래 위주 영상(춤보다 노래에
초점, 얼굴 필터로 왜곡)을 제외하고, 조회수 컷라인(약 80만 이상)을 넘긴 실사 1~3인
클립만 남겼다. 그중 1개(zUv5uklv-RE, 1,500만 뷰)는 QC에서 `trim_motion` 유지 비율이
25%로 떨어져(콜드스타트/컷 편집 아티팩트로 추정) 제외했다 — 조회수가 높아도 추출
품질이 깨지면 그대로 버린다는 원칙(youtube_dance_analysis.md 10번 참고)을 그대로 적용.

후보를 더 찾아 n=4 -> 6으로 확장(2026-07-10): "Mirrored" 댄스 연습 영상과 3인조
크리에이터(A3 BEATS, 3인이 HUNTR/X 세 멤버를 각자 맡아 옥상에서 촬영) 2건을 추가.
둘 다 QC(유지 프레임 비율 90%대) 통과.

최종 6건 (2026-07-10 기준 조회수, 내림차순):
  fB6pofm4klQ   33,938,808  솔로_반짝의상
  1OPKKCIq3jA    4,360,564  솔로_리뷰워터마크
  hPV242q5mGA    1,541,065  3인_커튼방
  pRgRq8u_uFo    1,346,619  듀오_연습실(미러링)
  4byOkaKIJrc    1,105,658  3인_옥상(A3BEATS)
  fxkI8QVKoL4      881,089  솔로_원곡비교
  (zUv5uklv-RE   15,057,840  QC 실패로 제외)

실행: python analyze_golden_dance.py
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
    "fB6pofm4klQ-Huntrix - Golden(dance cover) ｜ KPop Demon Hunters.webm": "솔로_반짝의상(3390만)",
    "1OPKKCIq3jA-HUNTRIX ‘Golden’ Dance Cover Challenge (from Kpop .webm": "솔로_리뷰워터마크(436만)",
    "hPV242q5mGA-HUNTRIX - Golden Dance Cover ｜ Kpop Demon Hunters .webm": "3인_커튼방(154만)",
    "pRgRq8u_uFo-Huntrix - Golden Dance Practice Mirrored (Kpop Dem.webm": "듀오_연습실(135만)",
    "4byOkaKIJrc-Golden - HUNTR⧸X dance cover ｜ K-POP DEMON HUNTERS.webm": "3인_옥상A3BEATS(111만)",
    "fxkI8QVKoL4-Golden - HUNTR⧸X from KPOP DEMON HUNTERS dance cov.webm": "솔로_원곡비교(88만)",
}


def main() -> None:
    files = sorted(Path("sample_dance_golden").glob("*"))
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
    print("\n전체 쌍 (가까운 순):")
    for a, b, d in pairs:
        print(f"  {a} <-> {b} : {d:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"Golden 챌린지(조회수 상위 {n}건) — 3관절 평균 DTW 거리(경로 길이 정규화)", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_golden_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_golden_dtw.png")


if __name__ == "__main__":
    main()
