"""개인 특징이 동작을 넘어가나 — 조깅(M06)·걷기(M01)·지친 걷기(M07) 사이 교차 검증 (2D, MediaPipe).

질문: "조깅에서 팔을 펴고 뛴 사람이 걸을 때도 팔을 펴나?" — 스타일을 다른 동작에 입히려면 이게 성립해야 한다
(9/29 회의 이태희: "조깅에서 뽑은 특징은 조깅 계열에만 유효"라는 의견을 데이터로 확인).

특징은 크기와 무관한 **각도**만 쓴다 — 키·카메라 거리·원근에 안 흔들리고, 몸통 길이로 나눈 길이 특징처럼
같은 몸이라 저절로 맞는(체형) 성분이 덜 섞인다. 비교용으로 팔다리 길이 비율(체형)도 같이 잰다.
  팔꿈치: 어깨-팔꿈치-손목 각(180°=폄)         무릎: 골반-무릎-발목 각의 평균
  팔 벌림: 몸통선과 위팔 사이 각                 허벅지 들기: 몸통선과 허벅지 사이 각의 범위(p95−p5)
  상체 기울기: 수직 대비 몸통 각                 고개: 목→코 선과 몸통선 사이 각

사용법 (repo 루트, pca_pilot.py 캐시 재사용 — 없는 영상만 새로 관절 추출):
  .venv/bin/python cross_motion.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import _smooth  # noqa: E402
from build_icc_features import _trim_window  # noqa: E402
from evaluate_icc import icc_3_1  # noqa: E402
import pca_pilot as Q  # noqa: E402

MOTIONS = ("M06", "M01", "M07")
NAMES = {"M06": "조깅", "M01": "걷기", "M07": "지친 걷기"}
# M06은 P005(discarded)·P009(화면 이탈) 제외, M07은 P003 미촬영
PEOPLE = "P001,P002,P003,P004,P006,P007,P008,P010,P011,P012,P013,P014".split(",")
FEATS = ("팔꿈치", "무릎", "팔 벌림", "허벅지 들기", "상체 기울기", "고개")


def takes(p: str, m: str) -> list[Path]:
    return sorted(Q.RAW.glob(f"{p}_S1_{m}_*T*_CAM_F.mov"))


def angles(rec: dict) -> dict[str, float]:
    d = Q.dyn8(rec)["dyn"]
    win = _trim_window(np.mean([d[j]["speed"] for j in ("오른엉덩이", "왼엉덩이")], axis=0))
    xy = np.stack([_smooth(rec["xy"][:, j].astype(float)) for j in range(33)], axis=1)[win]
    g = lambda j: xy[:, j]
    mid = lambda a, b: (g(a) + g(b)) / 2

    def ang(u, v):
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    hip, sh = mid(23, 24), mid(11, 12)
    trunk_down = hip - sh  # 어깨→골반
    elbow = np.mean([ang(g(s) - g(e), g(w) - g(e)).mean() for s, e, w in ((11, 13, 15), (12, 14, 16))])
    knee = np.mean([ang(g(h) - g(k), g(a) - g(k)).mean() for h, k, a in ((23, 25, 27), (24, 26, 28))])
    abduct = np.mean([ang(trunk_down, g(e) - g(s)).mean() for s, e in ((11, 13), (12, 14))])
    thigh = np.mean([np.ptp(np.percentile(ang(trunk_down, g(k) - g(h)), [5, 95])) for h, k in ((23, 25), (24, 26))])
    lean = np.degrees(np.arctan2(np.abs(sh[:, 0] - hip[:, 0]), hip[:, 1] - sh[:, 1])).mean()
    head = ang(sh - hip, g(0) - sh).mean()
    L = np.linalg.norm(sh - hip, axis=1)
    limb = {"위팔": (np.linalg.norm(g(11) - g(13), axis=1) + np.linalg.norm(g(12) - g(14), axis=1)) / 2 / L,
            "허벅지": (np.linalg.norm(g(23) - g(25), axis=1) + np.linalg.norm(g(24) - g(26), axis=1)) / 2 / L,
            "정강이": (np.linalg.norm(g(25) - g(27), axis=1) + np.linalg.norm(g(26) - g(28), axis=1)) / 2 / L}
    return {"팔꿈치": elbow, "무릎": knee, "팔 벌림": abduct, "허벅지 들기": thigh, "상체 기울기": lean, "고개": head,
            **{k: float(np.median(v)) for k, v in limb.items()}}


def main() -> None:
    cache = pickle.loads(Q.CACHE.read_bytes()) if Q.CACHE.exists() else {}
    per = {}  # (p, m) -> 테이크별 특징 목록
    try:
        for m in MOTIONS:
            for p in PEOPLE:
                fs = takes(p, m)
                if not fs:
                    continue
                for f in fs:
                    if str(f) not in cache:
                        cache[str(f)] = Q.extract(f)
                per[(p, m)] = [angles(cache[str(f)]) for f in fs]
    finally:
        Q.CACHE.write_bytes(pickle.dumps(cache))

    feats = FEATS + ("위팔", "허벅지", "정강이")
    mean_of = lambda p, m, k: float(np.mean([t[k] for t in per[(p, m)]]))
    report = {"within_icc": {}, "cross_r": {}, "cross_id": {}}

    print("① 동작 안에서 반복 촬영 일관성 ICC (테이크 2개 이상인 사람만)")
    for m in MOTIONS:
        ppl = [p for p in PEOPLE if (p, m) in per and len(per[(p, m)]) >= 2]
        row = {k: float(icc_3_1(np.array([[per[(p, m)][t][k] for t in range(2)] for p in ppl]))) for k in feats}
        report["within_icc"][m] = {"n": len(ppl), **row}
        print(f"  {NAMES[m]:5s}(n={len(ppl):2d}) " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))

    print("\n② 동작 사이 상관 r (사람별 평균값끼리) — 높으면 '조깅에서 그런 사람은 걸을 때도 그렇다'")
    pairs = [("M06", "M01"), ("M06", "M07"), ("M01", "M07")]
    for a, b in pairs:
        ppl = [p for p in PEOPLE if (p, a) in per and (p, b) in per]
        row = {k: float(np.corrcoef([mean_of(p, a, k) for p in ppl], [mean_of(p, b, k) for p in ppl])[0, 1]) for k in feats}
        report["cross_r"][f"{a}-{b}"] = {"n": len(ppl), **row}
        print(f"  {NAMES[a]}↔{NAMES[b]}(n={len(ppl):2d}) " + "  ".join(f"{k} {v:+.2f}" for k, v in row.items()))

    print("\n③ 동작을 넘는 식별: A동작 프로필(사람별 평균, 동작 안에서 표준화)로 B동작 테이크가 누구인지 맞히기")
    for a, b in pairs + [(y, x) for x, y in pairs]:
        ppl = [p for p in PEOPLE if (p, a) in per and (p, b) in per]
        for name, ks in (("각도 6개", FEATS), ("체형(팔다리 비율)", ("위팔", "허벅지", "정강이"))):
            A = np.array([[mean_of(p, a, k) for k in ks] for p in ppl])
            mu, sd = A.mean(0), A.std(0)
            A = (A - mu) / sd
            Bt = [(i, np.array([t[k] for k in ks])) for i, p in enumerate(ppl) for t in per[(p, b)]]
            Bm = np.array([v for _, v in Bt])
            Bz = (Bm - Bm.mean(0)) / Bm.std(0)  # B동작 안에서 표준화(동작 차이 제거)
            hit = np.mean([np.argmin(((A - Bz[j]) ** 2).sum(1)) == i for j, (i, _) in enumerate(Bt)])
            report["cross_id"][f"{a}->{b}:{name}"] = {"n": len(ppl), "acc": float(hit), "chance": 1 / len(ppl)}
            print(f"  {NAMES[a]}→{NAMES[b]}(n={len(ppl):2d}) {name}: {hit:.0%} (우연 {1 / len(ppl):.0%})")
    Path("reports/cross_motion.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
