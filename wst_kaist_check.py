"""카이스트 WST 축 조절 데모 영상으로 우리 WST 근사식(wst_pilot.py)을 검증.

영상: 안무 4개(CamelWalk·PepperSeed·Skate·WuTang) × {neutral, GT, Ours} × 축 {W,S,T} × 방향 {+,-}.
파일명 규칙 `{안무}_{GT|Ours}_{축}{-?}_{배율}x.mp4` — "-" 없음 = + 방향(Strong/Direct/Sudden)으로 가정.
막대인형 렌더(960x540, 30fps)에 MediaPipe를 그대로 돌리고, 같은 안무의 neutral 대비 변화율을 본다.
같은 골격·같은 카메라라 몸 크기 정규화는 불필요 — 원값(W95, T, S, F)을 쓴다.

기대: W+ → W95↑, S+ → S(직선성)↑, T+ → T(가속)·F(저크)↑, "-"는 반대.

사용법:
  .venv/bin/python wst_kaist_check.py /Users/enujes/Downloads/WST
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from wst_pilot import ALL, load, wst

CACHE = Path("reports/wst_kaist_cache.pkl")
DANCES = ["CamelWalk", "PepperSeed", "Skate", "WuTang"]
AXES = {"W": ("W95", "1x"), "S": ("S", "2x"), "T": ("T", "3x")}  # 축 → (우리 지표, 공식 배율)
METRICS = ("W95", "S", "T", "F")


def main(root: Path) -> None:
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    rows = []
    try:
        for d in DANCES:
            base = wst(load(str(root / f"{d}_neutral.mp4"), cache), ALL)
            for src in ("GT", "Ours"):
                for ax, (_, mult) in AXES.items():
                    for sign in ("+", "-"):
                        f = root / f"{d}_{src}_{ax}{'-' if sign == '-' else ''}_{mult}.mp4"
                        m = wst(load(str(f), cache), ALL)
                        rows.append({"dance": d, "src": src, "axis": ax, "sign": sign,
                                     **{k: 100 * (m[k] / base[k] - 1) for k in METRICS}})
    finally:
        CACHE.write_bytes(pickle.dumps(cache))
    Path("reports/wst_kaist_check.json").write_text(json.dumps(rows, indent=1))

    print("neutral 대비 변화율(%) — 행: 조절한 축/방향, 열: 우리 지표. 4개 안무 평균 [부호 맞은 안무 수/4]")
    for src in ("GT", "Ours"):
        print(f"\n== {src}")
        print(f"{'':6s}" + "".join(f"{k:>16s}" for k in METRICS))
        for ax, (target, _) in AXES.items():
            for sign in ("+", "-"):
                sel = [r for r in rows if r["src"] == src and r["axis"] == ax and r["sign"] == sign]
                cells = []
                for k in METRICS:
                    v = np.array([r[k] for r in sel])
                    want = 1 if sign == "+" else -1
                    hit = f"[{int((np.sign(v) == want).sum())}/4]" if k == target else ""
                    cells.append(f"{v.mean():+7.1f}{hit:>6s}")
                print(f"{ax}{sign:5s}" + "".join(f"{c:>16s}" for c in cells))

    # GT와 Ours가 같은 방향·크기로 움직이는지: 24조건(안무×축×방향)의 목표 지표 변화율 상관
    for ax, (target, _) in AXES.items():
        g = [r[target] for r in rows if r["src"] == "GT" and r["axis"] == ax]
        o = [r[target] for r in rows if r["src"] == "Ours" and r["axis"] == ax]
        print(f"GT↔Ours {ax}축({target}) 변화율 상관 r={np.corrcoef(g, o)[0, 1]:+.2f}, "
              f"평균 크기 GT {np.mean(np.abs(g)):.1f}% / Ours {np.mean(np.abs(o)):.1f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
