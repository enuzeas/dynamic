"""동작에 종속되지 않는 개인 특징 찾기 — ① 가만히 선 자세(M00)가 걷기·조깅 자세를 예측하나 ② 좌우 비대칭이 동작을 넘어가나.

3D(MediaPipe world, mp3d_extract.py)에서 몸 좌표계 기준 각도(style_compare.channels)를 쓴다 — 정면(서기)과
옆모습(걷기)처럼 카메라 방향이 달라도 같은 정의가 되도록. 사람 간 비교라 체형(뼈 길이)은 쓰지 않는다.

  자세 특징(평균): 상체 앞/옆 기울기, 고개 앞/옆, 팔 벌림, 팔 앞뒤, 팔꿈치, 무릎, 다리 벌림
  비대칭(왼쪽−오른쪽): 어깨 높이 차, 고개 옆 기울기, 상체 옆 기울기, 팔 벌림·팔꿈치·무릎·다리 벌림 차,
                     (움직이는 동작만) 팔 흔들림·허벅지 흔들림 폭의 로그비

motion_puzzle conda env: conda run --no-capture-output -n motion_puzzle python invariant_traits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style_compare as S
from evaluate_icc import icc_3_1

PEOPLE = S.PEOPLE
MOTIONS = {"M00": "서기", "M01": "걷기", "M06": "조깅", "M07": "지친 걷기"}
POSTURE = ["상체 앞기울기", "상체 옆기울기", "고개 앞", "고개 옆", "팔 벌림", "팔 앞뒤", "팔꿈치", "무릎", "다리 벌림"]
ASYM_STATIC = ["어깨 높이 차", "고개 옆기울기", "상체 옆기울기(비대칭)", "팔 벌림 차", "팔꿈치 차", "무릎 차", "다리 벌림 차"]
ASYM_DYN = ["팔 흔들림 비", "허벅지 흔들림 비"]


def standing(pts: dict, ch: dict) -> np.ndarray:
    """M00에서 T포즈가 아닌, 팔을 내리고 가만히 선 가장 긴 구간(프레임 마스크)."""
    low = (ch["Larm_front"] < 30) & (ch["Rarm_front"] < 30) & (np.abs(ch["Larm_sag"]) < 30) & (np.abs(ch["Rarm_sag"]) < 30)
    e = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
    runs = [(a, b) for a, b in zip(e[::2], e[1::2])]
    a, b = max(runs, key=lambda r: r[1] - r[0])
    m = np.zeros(len(low), bool)
    m[a:b] = True
    return m


def features(npz: Path, motion: str) -> dict:
    pts = S.mp_points(npz)[0]
    ch = S.channels(pts)
    m = standing(pts, ch) if motion == "M00" else np.ones(len(ch["trunk_sag"]), bool)
    mean = lambda k: float(np.mean(ch[k][m]))
    sd = lambda k: float(np.std(ch[k][m]))
    # 어깨 높이 차: 몸 좌표계 위 방향으로 왼어깨−오른어깨, 어깨 폭으로 나눔
    up = S.nrm(pts["neck"] - pts["pelvis"])
    sw = np.linalg.norm(pts["Lsh"] - pts["Rsh"], axis=-1)
    sh_diff = float(np.mean((((pts["Lsh"] - pts["Rsh"]) * up).sum(-1) / sw)[m]))
    f = {
        "상체 앞기울기": mean("trunk_sag"), "상체 옆기울기": abs(mean("trunk_front")),
        "고개 앞": mean("head_sag"), "고개 옆": abs(mean("head_front")),
        "팔 벌림": (mean("Larm_front") + mean("Rarm_front")) / 2, "팔 앞뒤": (mean("Larm_sag") + mean("Rarm_sag")) / 2,
        "팔꿈치": (mean("Lelbow") + mean("Relbow")) / 2, "무릎": (mean("Lknee") + mean("Rknee")) / 2,
        "다리 벌림": (mean("Lthigh_front") + mean("Rthigh_front")) / 2,
        "어깨 높이 차": sh_diff, "고개 옆기울기": mean("head_front"), "상체 옆기울기(비대칭)": mean("trunk_front"),
        "팔 벌림 차": mean("Larm_front") - mean("Rarm_front"), "팔꿈치 차": mean("Lelbow") - mean("Relbow"),
        "무릎 차": mean("Lknee") - mean("Rknee"), "다리 벌림 차": mean("Lthigh_front") - mean("Rthigh_front"),
    }
    if motion != "M00":
        f["팔 흔들림 비"] = float(np.log(sd("Larm_sag") / sd("Rarm_sag")))
        f["허벅지 흔들림 비"] = float(np.log(sd("Lthigh_sag") / sd("Rthigh_sag")))
    return f


def main() -> None:
    per = {}  # (p, m) -> [테이크별 특징]
    for p in PEOPLE:
        for m in MOTIONS:
            fs = sorted(S.MP3D.glob(f"{p}_S1_{m}_*T*.npz"))
            if fs:
                per[(p, m)] = [features(f, m) for f in fs]
    avg = lambda p, m, k: float(np.mean([t[k] for t in per[(p, m)]]))
    rep = {"icc": {}, "cross_r": {}, "cross_id": {}}

    print("① 동작 안 반복 일관성 ICC (2테이크 이상)")
    for m in ("M01", "M06", "M07"):
        ppl = [p for p in PEOPLE if len(per.get((p, m), [])) >= 2]
        if len(ppl) < 4:
            continue
        ks = POSTURE + ASYM_STATIC + ASYM_DYN
        row = {k: float(icc_3_1(np.array([[per[(p, m)][t][k] for t in range(2)] for p in ppl]))) for k in ks}
        rep["icc"][m] = {"n": len(ppl), **row}
        print(f"  {MOTIONS[m]}(n={len(ppl)}): " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))

    pairs = [("M00", "M01"), ("M00", "M06"), ("M06", "M01"), ("M01", "M07"), ("M06", "M07")]
    print("\n② 동작 사이 상관 r (사람별 평균)")
    for a, b in pairs:
        ppl = [p for p in PEOPLE if (p, a) in per and (p, b) in per]
        ks = POSTURE + ASYM_STATIC + (ASYM_DYN if "M00" not in (a, b) else [])
        row = {k: float(np.corrcoef([avg(p, a, k) for p in ppl], [avg(p, b, k) for p in ppl])[0, 1]) for k in ks}
        rep["cross_r"][f"{a}-{b}"] = {"n": len(ppl), **row}
        print(f"  {MOTIONS[a]}↔{MOTIONS[b]}(n={len(ppl)}): " + "  ".join(f"{k} {v:+.2f}" for k, v in row.items()))

    print("\n③ 동작을 넘는 식별: A동작 프로필로 B동작 테이크의 주인 맞히기 (동작 안 표준화)")
    for a, b in pairs + [("M01", "M06"), ("M01", "M00")]:
        ppl = [p for p in PEOPLE if (p, a) in per and (p, b) in per]
        for name, ks in (("자세", POSTURE), ("비대칭", ASYM_STATIC), ("자세+비대칭", POSTURE + ASYM_STATIC)):
            A = np.array([[avg(p, a, k) for k in ks] for p in ppl])
            A = (A - A.mean(0)) / (A.std(0) + 1e-9)
            Bt = [(i, [t[k] for k in ks]) for i, p in enumerate(ppl) for t in per[(p, b)]]
            Bm = np.array([v for _, v in Bt])
            Bz = (Bm - Bm.mean(0)) / (Bm.std(0) + 1e-9)
            hit = float(np.mean([np.argmin(((A - Bz[j]) ** 2).sum(1)) == i for j, (i, _) in enumerate(Bt)]))
            rep["cross_id"][f"{a}->{b}:{name}"] = {"n": len(ppl), "acc": hit}
            print(f"  {MOTIONS[a]}→{MOTIONS[b]}(n={len(ppl)}) {name}: {hit:.0%} (우연 {1 / len(ppl):.0%})")
    (S.ROOT / "reports" / ("invariant_traits.json" if S.SRC == "mp3d" else f"invariant_traits_{S.SRC}.json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
