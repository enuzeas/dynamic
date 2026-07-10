"""유튜브 쇼츠 '애플(Apple, Charli XCX)' 댄스 챌린지 커버 6개 — 킥드럼 챌린지와 다른 안무로 교차검증.

analyze_youtube_dance.py(킥드럼 챌린지)와 같은 방법론을 다른 챌린지에 적용해,
"같은 안무·다른 사람" 조건에서의 DTW 거리 패턴이 특정 챌린지에 국한된 우연이 아닌지 확인한다.
sample_dance_apple/ 폴더 영상은 전부 켈리 헤이어가 만든 애플 챌린지 안무
(팔 교차·체스트 팝·엉덩이 탭·스플릿·운전 동작 등)를 따라 춘 것들이다.

각 영상이 크리에이터/등장인물 1인당 1개뿐이라 반복(항상성) 검증은 안 되고,
서로 다른 사람 간 거리 분포(고유성)만 본다. 일부 영상은 2~3인이 함께 등장한다
(킥드럼 챌린지 데이터셋에서도 다인원 클립이 섞여 있었고 문제 없이 처리됐다).

실행: python analyze_apple_dance.py
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
    "EIJZCfGmEpg-Charli XCX Apple Dance #shorts.webm": "solo_A",
    "IC0tq6n1zkY-Charli XCX & Troye Sivan ATE the Apple Dance 🍏.webm": "Charli+Troye+1(3인)",
    "MBRvlF7LgFU-Viral Charli XCX Apple Dance Challenge Tutorial #c.webm": "튜토리얼_A(자막)",
    "MNStAqiH--w-Rosé excitedly dances to Charli XCX's Apple dance .webm": "로제+1(2인)",
    "piSwPc3JhJw-Charli XCX Apple Dance Challenge Tutorial #tiktokd.webm": "튜토리얼_B(자막)",
    "tfI7xocwLe0-[MIRRORED] Charli xcx - Apple 🍏 Tiktok Dance Chall.webm": "solo_B(미러링)",
}


def main() -> None:
    files = sorted(Path("sample_dance_apple").glob("*"))
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

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"애플 챌린지 {n}건 — 3관절(손목·팔꿈치·어깨) 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_apple_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_apple_dtw.png")


if __name__ == "__main__":
    main()
