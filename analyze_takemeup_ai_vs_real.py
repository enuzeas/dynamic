"""'Take Me Up' 챌린지 — 실사 클러스터 vs AI 의심 클러스터 비교.

analyze_takemeup_dance.py(n=2)의 후속. 사용자가 유튜브의 "동일 오디오 사용 영상"
리믹스 페이지(YouTube 자체 검색으로는 못 찾던 목록, 브라우저로 직접 확인해 링크
16개를 전달)를 통해 후보를 대폭 확보했다. 프레임 확인 중 뚜렷한 패턴을 발견했다 —
후보의 절반 가까이가 "이국적 배경(에펠탑·해변 리조트·산 전망 발코니) + 비슷한 인상의
미인 얼굴 + 똑같은 팔 들어올리는 포즈"를 반복하는 템플릿 특징을 보였고, 일부는
제목에 아예 "AI 댄스"라고 자칭했다("두진위" 태그도 여러 챌린지에 반복 등장하는
동일 계열로 추정). 사용자가 최초에 보낸 영상(tFePzSW3AVE)의 자체 캡션도
"ai인지 사람인지 헷깔림"이었다 — 즉 n=2 파일럿에 이미 의심 클러스터가 섞여 있었을
가능성이 있다.

이 프로젝트의 핵심 명제(BIO-IP)가 "AI는 사람의 움직임을 복제 못 한다"는 것이므로,
이 우연한 발견은 실측으로 확인해볼 좋은 기회다. 분류는 프레임 육안 검토에만
근거한 것으로, 계정 신원이나 생성 도구를 확인한 사실은 아니다 — "의심" 수준이다.

REAL(8): 실제 공원·시골·주차장·행사장 배경, 자연스러운 모션 블러, "AFstarz"(실존 街댄스
크루로 추정) 태그, dj9IT2AC3iY(원래 n=2 파일럿의 실사 클립).
AISUSPECT(9): 이국적 배경 + 반복 포즈 템플릿, tFePzSW3AVE(사용자 제공 원본 포함).

실행: python analyze_takemeup_ai_vs_real.py
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
    "REAL_gV3yefAb1Ao.webm": "R_공원_서아",
    "REAL_BUZAGZG2ZsQ.webm": "R_밀밭_중국",
    "REAL_q-6X-Ektd9Q.webm": "R_공원_은빈",
    "REAL_CbbRIn1ibm4.webm": "R_행사장_나율",
    "REAL_MHkk9onWMuo.webm": "R_골목_중국",
    "REAL_vmoZIJg-X20.webm": "R_주차장_수빈(듀오)",
    "REAL_dj9IT2AC3iY.webm": "R_주방_에이프런",
    "REAL_WyPYv01TFmg.webm": "R_스튜디오_체리",
    "AISUSPECT_tFePzSW3AVE.webm": "AI_발코니_원본",
    "AISUSPECT_nxYBzzqf_nQ.webm": "AI_주방_민지",
    "AISUSPECT_71koL-3XxxE.webm": "AI_에펠탑_에이미",
    "AISUSPECT_1w4mBcLBTaY.webm": "AI_주방_중국1",
    "AISUSPECT_Lewk8lXv38k.webm": "AI_주방_중국2",
    "AISUSPECT_ekIMPnW3qbU.webm": "AI_스튜디오_두진위1",
    "AISUSPECT_SeDYhQhYcJ0.webm": "AI_에펠탑_두진위2",
    "AISUSPECT_Rp-S3JRYn8k.webm": "AI_산전망_자칭AI",
    "AISUSPECT_ahx0RIAnt8U.webm": "AI_리조트_두진위3",
}


def main() -> None:
    files = sorted(Path("sample_dance_takemeup_full").glob("*"))
    speeds: dict[str, dict[str, np.ndarray]] = {}
    for f in files:
        label = LABELS.get(f.name, f.stem)
        dyn = extract_joint_dynamics(str(f), joints=JOINTS)
        speeds[label] = {j: trim_motion(dyn[j]["speed"]) for j in JOINTS}
        lengths = ", ".join(f"{j}:{len(speeds[label][j])}" for j in JOINTS)
        print(f"추출 완료: {label} -> {lengths}")

    labels = list(speeds)
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            per_joint = [dtw(speeds[labels[i]][jt], speeds[labels[j]][jt]) for jt in JOINTS]
            D[i, j] = D[j, i] = float(np.mean(per_joint))

    real_idx = [i for i, l in enumerate(labels) if l.startswith("R_")]
    ai_idx = [i for i, l in enumerate(labels) if l.startswith("AI_")]

    within_real = np.array([D[i, j] for a, i in enumerate(real_idx) for j in real_idx[a+1:]])
    within_ai = np.array([D[i, j] for a, i in enumerate(ai_idx) for j in ai_idx[a+1:]])
    cross = np.array([D[i, j] for i in real_idx for j in ai_idx])

    print(f"\n실사 8건 내부 거리 — min {within_real.min():.4f} / mean {within_real.mean():.4f} / max {within_real.max():.4f} (n={len(within_real)}쌍)")
    print(f"AI의심 9건 내부 거리 — min {within_ai.min():.4f} / mean {within_ai.mean():.4f} / max {within_ai.max():.4f} (n={len(within_ai)}쌍)")
    print(f"실사<->AI의심 교차 거리 — min {cross.min():.4f} / mean {cross.mean():.4f} / max {cross.max():.4f} (n={len(cross)}쌍)")

    # AI의심 9건 중에서도 "이국적 배경+반복 포즈 템플릿" 특징이 뚜렷한 7건만 따로 —
    # 같은 모션소스를 각기 다른 배경/얼굴에 입힌 AI 생성물이라면 이 7건끼리 유독 가까울 것이라는 가설 검증
    template7 = [
        "AI_주방_중국1", "AI_에펠탑_에이미", "AI_주방_중국2", "AI_산전망_자칭AI",
        "AI_에펠탑_두진위2", "AI_리조트_두진위3", "AI_스튜디오_두진위1",
    ]
    others = [l for l in labels if l not in template7]
    t_idx = [labels.index(l) for l in template7]
    o_idx = [labels.index(l) for l in others]
    within_t = np.array([D[i, j] for a, i in enumerate(t_idx) for j in t_idx[a + 1:]])
    within_o = np.array([D[i, j] for a, i in enumerate(o_idx) for j in o_idx[a + 1:]])
    cross_to = np.array([D[i, j] for i in t_idx for j in o_idx])
    print(f"\n[템플릿 의심 7건 vs 나머지 10건(실사8+AI2)]")
    print(f"템플릿7 내부 — min {within_t.min():.4f} / mean {within_t.mean():.4f} / max {within_t.max():.4f} (n={len(within_t)}쌍)")
    print(f"나머지10 내부 — min {within_o.min():.4f} / mean {within_o.mean():.4f} / max {within_o.max():.4f} (n={len(within_o)}쌍)")
    print(f"템플릿7<->나머지10 교차 — min {cross_to.min():.4f} / mean {cross_to.mean():.4f} / max {cross_to.max():.4f} (n={len(cross_to)}쌍)")

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    im = ax.imshow(D, cmap="plasma_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#D4DEEE")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7, color="#D4DEEE")
    ax.set_title("Take Me Up — 실사(R_) vs AI의심(AI_) 3관절 평균 DTW 거리", color="#D4DEEE", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig("sample_dance_takemeup_ai_vs_real_dtw.png", dpi=150, bbox_inches="tight", facecolor="#080A10")
    print("\n저장 완료: sample_dance_takemeup_ai_vs_real_dtw.png")


if __name__ == "__main__":
    main()
