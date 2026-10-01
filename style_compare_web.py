"""style_compare.py 결과를 비교 웹페이지(style_compare.html 틀)에 넣는다.

클립마다 관절 17개의 3D 위치를 30fps로 줄여 넣는다. 보기 좋게 하려고
  - 골반을 수평 원점에 고정(제자리 걷기처럼), 발 최저점을 바닥(y=0)에
  - 몸이 향한 방향(골반 좌우선 기준, 1초 평활)을 +Z로 돌려 고정
  - 다리 길이(골반→무릎→발목)를 1로 맞춤 — 사람·골격 사이 크기 차이 제거
motion_puzzle conda env에서: conda run --no-capture-output -n motion_puzzle python style_compare_web.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import style_compare as S

KEYS = ["pelvis", "Lhip", "Lknee", "Lank", "Ltoe", "Rhip", "Rknee", "Rank", "Rtoe", "neck", "head",
        "Lsh", "Lel", "Lwr", "Rsh", "Rel", "Rwr"]


def pack(pts: dict, fps: float) -> dict:
    step = max(1, int(round(fps / 30)))
    P = np.stack([pts[k] for k in KEYS], axis=1)[::step].astype(float)  # (T,17,3)
    leg = np.median(np.linalg.norm(P[:, 1] - P[:, 2], axis=1) + np.linalg.norm(P[:, 2] - P[:, 3], axis=1))
    P = P / leg
    across = P[:, 1] - P[:, 5]
    across[:, 1] = 0
    across = gaussian_filter1d(across, sigma=max(1, 30 / 6), axis=0)
    yaw = np.arctan2(across[:, 2], across[:, 0])  # 좌우선이 +X를 향하도록 → 앞은 +Z
    c, s = np.cos(yaw), np.sin(yaw)
    pel = P[:, :1].copy()
    pel[..., 1] = 0
    P = P - pel
    x, z = P[..., 0].copy(), P[..., 2].copy()
    P[..., 0] = c[:, None] * x + s[:, None] * z
    P[..., 2] = -s[:, None] * x + c[:, None] * z
    P[..., 1] -= np.percentile(P[:, [3, 4, 7, 8], 1], 1)
    return {"fps": fps / step, "x": [round(float(v), 3) for v in P.ravel()], "n": len(P)}


def main() -> None:
    rep = json.loads((S.OUT / "style_compare.json").read_text())
    canim, cnames, cft = S.BVH.load(str(S.CONTENT["walk"]))
    data = {"keys": KEYS, "metrics": list(S.METRICS), "base": rep["base"], "eval": rep["eval"], "people": {},
            "content": pack(S.bvh_points(canim, cnames), 1 / cft)}
    for p in S.PEOPLE:
        jog_pts, _, jfps = S.mp_points(S.MP3D / f"{p}_S1_M06_T01.npz")
        walk_f = sorted(S.MP3D.glob(f"{p}_S1_M01_*T*.npz"))[0]
        walk_pts, _, wfps = S.mp_points(walk_f)
        off, _, oft = S.BVH.load(str(S.OUT / "offset" / f"{p}_walk.bvh"))
        mpz, mnames, mft = S.BVH.load(str(S.OUT / "puzzle" / f"Style_{p}_jog_Content_35_06_fixed.bvh"))
        a3, n3, f3 = S.BVH.load(str(S.OUT / "aberman3d" / f"{p}_jog" / "fixed.bvh"))
        a2, n2, f2 = S.BVH.load(str(S.OUT / "aberman2d" / f"{p}_jog" / "fixed.bvh"))
        ada, _, fa = S.BVH.load(str(S.OUT / "adain" / f"{p}_jog.bvh"))
        data["people"][p] = {
            "clips": {"jog": pack(jog_pts, jfps), "puzzle": pack(S.bvh_points(mpz, mnames), 1 / mft),
                      "adain": pack(S.bvh_points(ada, S.N31), 1 / fa),
                      "aberman3d": pack(S.bvh_points(a3, n3), 1 / f3), "aberman2d": pack(S.bvh_points(a2, n2), 1 / f2),
                      "offset": pack(S.bvh_points(off, S.N31), 1 / oft), "walk": pack(walk_pts, wfps)},
            "m": {k: rep[k][p] for k in ("jog", "puzzle", "aberman3d", "aberman2d", "offset", "adain", "walk")},
        }
    same = json.loads((S.OUT / "style_compare_same.json").read_text())
    data["same"] = {"eval": same["eval"], "people": {}}
    for p in same["people"]:
        w1, w2 = sorted(S.MP3D.glob(f"{p}_S1_M01_*T*.npz"))[:2]
        p1, _, f1 = S.mp_points(w1)
        p2, _, f2 = S.mp_points(w2)
        off, _, oft = S.BVH.load(str(S.OUT / "offset" / f"{p}_walk_from_walk.bvh"))
        mpz, mnames, mft = S.BVH.load(str(S.OUT / "puzzle" / f"Style_{p}_walk_Content_35_06_fixed.bvh"))
        a3, n3, g3 = S.BVH.load(str(S.OUT / "aberman3d" / f"{p}_walk" / "fixed.bvh"))
        a2, n2, g2 = S.BVH.load(str(S.OUT / "aberman2d" / f"{p}_walk" / "fixed.bvh"))
        ada, _, ga = S.BVH.load(str(S.OUT / "adain" / f"{p}_walk.bvh"))
        data["same"]["people"][p] = {
            "clips": {"jog": pack(p1, f1), "puzzle": pack(S.bvh_points(mpz, mnames), 1 / mft),
                      "adain": pack(S.bvh_points(ada, S.N31), 1 / ga),
                      "aberman3d": pack(S.bvh_points(a3, n3), 1 / g3), "aberman2d": pack(S.bvh_points(a2, n2), 1 / g2),
                      "offset": pack(S.bvh_points(off, S.N31), 1 / oft), "walk": pack(p2, f2)},
            "m": {"jog": same["src"][p], "puzzle": same["puzzle"][p], "aberman3d": same["aberman3d"][p],
                  "aberman2d": same["aberman2d"][p], "offset": same["offset"][p], "adain": same["adain"][p], "walk": same["held"][p]},
        }
    html = (S.ROOT / "style_compare.html").read_text().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    dst = S.OUT / "style_compare.html"
    dst.write_text(html)
    print("저장:", dst, f"({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
