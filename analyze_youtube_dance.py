"""유튜브 쇼츠 킥드럼(베이스) 챌린지 커버댄스 11개 — 같은 안무, 다른 사람 비교.

sample_dance/ 폴더의 영상들은 전부 같은 챌린지 안무(주먹 들어올리는 동작 포함)를
서로 다른 크리에이터가 각자 촬영해 올린 것들이다. groupdance.md가 말한
"안무로 콘텐츠가 통제된 자연 실험" 조건을 실제 인터넷 데이터로 채운 첫 사례.

각 영상이 크리에이터 1인당 1개뿐이라 반복(항상성) 검증은 안 되고,
서로 다른 사람 간 거리 분포(고유성)만 본다.

실행: python analyze_youtube_dance.py
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

# 파일명(크리에이터 핸들 기반) -> 표시 라벨. 영상 자체에 공개돼 있는 이름/핸들만 쓴다.
LABELS = {
    "@_kpop_sura_djsura_dance_challenge_trending_fyp_meme-(1920p30).mp4": "sura_A(스튜디오)",
    "_-(1920p60).mp4": "SOONIGROUP(버스킹·안무창시자)",
    "dancechallenge_onepickent-(1280p60).mp4": "onepickent",
    "fitness_fitnessmotivation_50-(1920p30).mp4": "fitness_50",
    "kickdrum_challenge_dance_sura_queenbee_yunamong-(1920p60).mp4": "sura_B(스튜디오)",
    "shorts-(3840p30).webm": "shorts_unknown",
    "4wTpIazbp-U-Kick Drum Bass 🔊.webm": "3인_기타드럼밈",
    "DkARMCiaoQ4-Queen Bee Kick Drum Bass Dance Cover.webm": "2인_커버(마스크)",
    "nE8KxiBU4Us-Kick drum bass 🥰💃#dance #masakakidsafricana #short.webm": "masaka_kids_africana",
    "sF9W3Gcte-o-Kick drum bass 🥁.webm": "3인_한옥마당",
    "xaRc8xzkHhw-가은이가 추는 Kick drum bass~Kick Kick drum base~🦵🥁🎸 #da.webm": "가은",
}


def main() -> None:
    files = sorted(Path("sample_dance").glob("*"))
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

    sura_pair = next(
        (p for p in pairs if any("sura_A" in x for x in (p[0], p[1])) and any("sura_B" in x for x in (p[0], p[1]))),
        None,
    )
    if sura_pair:
        rank = pairs.index(sura_pair) + 1
        print(f"\nsura_A <-> sura_B 거리: {sura_pair[2]:.4f} (전체 {len(pairs)}쌍 중 {rank}번째로 가까움)")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8, color="#D4DEEE")
    ax.set_title(f"킥드럼 챌린지 {n}인 — 3관절(손목·팔꿈치·어깨) 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_dtw.png")


if __name__ == "__main__":
    main()
