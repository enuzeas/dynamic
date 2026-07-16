"""유튜브 쇼츠 'APT.(로제·브루노 마스)' 댄스 챌린지 커버 11개 — 세 번째 챌린지로 교차검증 확장.

analyze_youtube_dance.py(킥드럼)·analyze_apple_dance.py(애플)에 이은 세 번째 챌린지.
APT.는 유튜브가 공식 Shorts 챌린지 캠페인으로 밀었을 만큼 커버 수가 압도적으로 많아
(Shorty Awards 수상작), 이번엔 후보 20개를 검토해 11개까지 추렸다 — 앞의 두 챌린지(6개씩)보다 큰 표본.

후보 중 제외한 것들과 사유:
- 3인 이상 그룹(농구 유니폼 7인조, 6인 라인업 등): 단일 인물 추적 가정이 흔들려 제외.
- 분할 화면(VER1/VER2 좌우 동시 비교): 한 프레임에 서로 다른 두 안무가 같이 잡혀 제외.
- 수중 촬영: 굴절·기포로 MediaPipe 왜곡 위험 + 실제 안무 수행 여부 불확실해 제외.
- 0.5배속 명시 클립: 실제 속도가 아니라서 다른 클립과 속도 비교가 무의미해 제외.
- 곡·안무 매칭이 불확실한 제네릭 영상: 챌린지 자체를 못 알아봐서 제외.

우연한 발견: "커튼_A" 3개(튜토리얼·템포·결과)가 같은 배경(빨간 커튼)·같은 스타일 자막으로,
동일 크리에이터의 반복 게시물로 추정된다 — 킥드럼 챌린지의 sura_A/sura_B와 같은
"동일 인물 반복" 신호를 이번엔 2쌍이 아니라 3개 클립(3쌍)으로 얻은 셈이다(항상성 참고용).

실행: python analyze_apt_dance.py
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

# 파일명 -> 표시 라벨. 영상 자체에 공개돼 있는 이름/맥락만 쓴다.
LABELS = {
    "01Wp7Bp6N14-APT. Dance Challenge Results 🪩 Viral Dance Trend 2.webm": "커튼_A(결과)",
    "7ts7UvlBwOo-APT Dance Challenge Pt 2! #TWINS #rosé #brunomars .webm": "듀오_아이들(twins)",
    "9CQjUz9fVXI-Viral Dance Challenge of 2024 #apt #aptchallenge D.webm": "듀오_스튜디오(aanya)",
    "ARW9YbBeRs8-APT Dance Challenge with #Zephanie & #DylanMenor #.webm": "듀오_필리핀(zephanie)",
    "GuJpK6pLDUQ-APT. Dance Tutorial 🪩 Viral Dance Challenge 2024.webm": "커튼_A(튜토리얼)",
    "KidQow3_dpY-APT. Dance Tutorial (Tempo) 🎉 Viral Dance Challeng.webm": "커튼_A(템포)",
    "_BYFgUDdbkY-I wanted to challenge my boyfriend and told him to.webm": "듀오_커플(모방)",
    "jmY3OM_rrXM-#apt #dance #tutorial #challenge #shorts #youtubes.webm": "solo_옥상",
    "ofBD5BLaBMQ-APT. Dance Challenge in public! ｜ ROSÉ, Bruno Mars.webm": "듀오_공원(모자)",
    "pSzQE5dugrY-WE FINALLY DID THE OTHER APT. DANCE ROSÉ & Bruno M.webm": "듀오_커플(핑크)",
    "vhgDNOa-Sss-APT. Dance Challenge ｜ Rosé, Bruno Mars.webm": "solo_현관",
}


def main() -> None:
    files = sorted(Path("sample_dance_apt").glob("*"))
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
            # dynamic_id.py의 _distance_to_id와 동일한 방식 — 관절별 DTW 거리의 평균
            per_joint = [dtw(speeds[labels[i]][jt], speeds[labels[j]][jt]) for jt in JOINTS]
            D[i, j] = D[j, i] = float(np.mean(per_joint))

    off_diag = D[np.triu_indices(n, k=1)]
    print(f"\n쌍별 DTW 거리 — min {off_diag.min():.4f} / mean {off_diag.mean():.4f} / max {off_diag.max():.4f}")

    pairs = [(labels[i], labels[j], D[i, j]) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: p[2])
    print("\n가장 가까운 3쌍 (구별이 가장 어려운 쌍):")
    for a, b, d in pairs[:3]:
        print(f"  {a} <-> {b} : {d:.4f}")
    print("가장 먼 3쌍 (가장 뚜렷하게 다른 쌍):")
    for a, b, d in pairs[-3:]:
        print(f"  {a} <-> {b} : {d:.4f}")

    # 커튼_A 3개 = 동일 크리에이터 추정 클립들 — 이 안에서의 쌍별 거리가
    # 서로 다른 사람 간 거리 분포에서 어디쯤 위치하는지 확인 (항상성 참고 신호)
    curtain_pairs = [p for p in pairs if p[0].startswith("커튼_A") and p[1].startswith("커튼_A")]
    if curtain_pairs:
        print("\n커튼_A(동일 크리에이터 추정) 내부 쌍:")
        for a, b, d in curtain_pairs:
            rank = pairs.index((a, b, d)) + 1
            print(f"  {a} <-> {b} : {d:.4f} (전체 {len(pairs)}쌍 중 {rank}번째로 가까움)")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"APT 챌린지 {n}건 — 3관절(손목·팔꿈치·어깨) 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_apt_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_apt_dtw.png")


if __name__ == "__main__":
    main()
