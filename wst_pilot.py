"""트랙 1 — WST(Weight·Space·Time) 역추출 ICC 파일럿.

카이스트 WST 축을 "과장"이 아니라 사람별 비교 지표로 쓰자는 결정(2026-09 회의)에 맞춰,
기존 P001~P007 영상에서 W·S·T 근사값을 뽑고 track1_icc_pilot.md와 같은 ICC(3,1)로
사람 구별력을 본다. 카이스트 쪽 추출기와는 별개인 우리 쪽 근사다.

정의는 Larboulette & Gibet(2015) "A review of computable expressive descriptors of human
motion"의 Laban Effort 계산식을 2D(MediaPipe 정규화 좌표)로 옮긴 것:
  W(Weight) = 트림 구간 운동에너지 Σ_j |v_j|² 의 최대값 (강함 vs 가벼움)
  T(Time)   = 트림 구간 가속도 크기 Σ_j |a_j| 의 평균 (급작 vs 지속)
  S(Space)  = 경로 직선성 = 직선거리 / 경로길이, 0.5초 창 평균 (직접 vs 우회)
              — 조깅처럼 주기적인 동작은 클립 전체로 재면 거의 0이 되므로 창으로 자른다.
  F(Flow)   = 트림 구간 벡터 저크 크기 Σ_j |j_j| 의 평균 (끊김 vs 매끄러움)
  baseline  = 기존 파일럿과 같은 mean_speed (재현 확인용)

사용법 (repo 루트, mediapipe 설치된 venv):
  .venv/bin/python wst_pilot.py            # 결과 표 출력 + reports/wst_pilot.json
  .venv/bin/python wst_pilot.py --combine  # 축 합산 점수 ICC / 다축 벡터 식별률
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import extract_joint_dynamics, LANDMARK_INDEX, _smooth  # noqa: E402
from demo import trim_motion  # noqa: E402
from build_icc_features import JOINT_PRESETS, _trim_window  # noqa: E402
from evaluate_icc import icc_3_1  # noqa: E402

ALL = tuple(LANDMARK_INDEX)
# (동작, 테이크 수, 참가자, 기존 파일럿 최선 관절 프리셋) — track1_icc_pilot.md §2와 동일 조건
RUNS = [
    ("M02", 3, "P001,P002,P003,P004,P005,P006,P007", "single"),
    ("M03", 2, "P001,P002,P003,P004,P005,P006,P007", "hands"),
    ("M04", 3, "P001,P002,P003,P004,P005,P006,P007", "hips"),
    ("M06", 2, "P001,P002,P003,P004,P006,P007", "hips"),  # P005 제외 — discarded_takes
]
CACHE = Path("reports/wst_pilot_cache.pkl")  # ponytail: 영상당 MediaPipe 1회만, 지우면 재추출
SPACE_WIN_S = 0.5
METRICS = ("W", "W95", "T", "S", "F", "baseline", "W_b", "W95_b", "T_b", "F_b", "baseline_b")


def _fps(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def load(path: str, cache: dict) -> dict:
    if path not in cache:
        cache[path] = {"dyn": extract_joint_dynamics(path, ALL), "fps": _fps(path)}
    return cache[path]


def directness(pos: np.ndarray, win: int) -> float:
    """창별 직선거리/경로길이의 평균 (1=완전 직선, 0=제자리 왕복)."""
    vals = []
    for s in range(0, len(pos) - win + 1, max(1, win // 2)):
        seg = pos[s:s + win]
        path = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
        if path > 1e-9:
            vals.append(np.linalg.norm(seg[-1] - seg[0]) / path)
    return float(np.mean(vals)) if vals else float("nan")


def wst(rec: dict, joints: tuple[str, ...]) -> dict[str, float]:
    dyn, fps = rec["dyn"], rec["fps"]
    # 트림 구간은 관절 평균 속도 기준 하나로 통일 — 관절마다 다르게 자르면 Σ가 안 맞는다
    win = _trim_window(np.mean([dyn[j]["speed"] for j in joints], axis=0))
    vel = {j: np.gradient(dyn[j]["pos"], axis=0)[win] * fps for j in joints}
    acc = {j: _smooth(np.gradient(np.gradient(dyn[j]["pos"], axis=0), axis=0) * fps * fps)[win] for j in joints}
    jrk = {j: _smooth(np.gradient(np.gradient(np.gradient(dyn[j]["pos"], axis=0), axis=0), axis=0) * fps**3)[win] for j in joints}
    energy = sum((vel[j] ** 2).sum(axis=1) for j in joints)
    accmag = sum(np.linalg.norm(acc[j], axis=1) for j in joints)
    w = max(3, int(round(SPACE_WIN_S * fps)))
    out = {
        "W": float(energy.max()),
        "W95": float(np.percentile(energy, 95)),  # 추적 튐 한 프레임에 max가 끌려가는지 확인용
        "T": float(accmag.mean()),
        "S": float(np.mean([directness(dyn[j]["pos"][win], w) for j in joints])),
        # F(Flow): 벡터 저크 크기 평균 — 매끄러움(free) vs 끊김(bound). 기존 mean_jerk는
        # 스칼라 속도를 미분한 값이라 방향 전환이 빠지는데, 이건 위치 벡터를 세 번 미분한다.
        "F": float(sum(np.linalg.norm(jrk[j], axis=1) for j in joints).mean()),
        "baseline": float(np.mean([np.mean(trim_motion(dyn[j]["speed"])) for j in joints])),
    }
    # 몸 크기 정규화(_b): 좌표가 프레임 대각선 기준이라 촬영 거리·체격이 속도에 섞인다.
    # 어깨중점-골반중점(몸통) 길이 중앙값으로 나눠, 그 교란을 뺀 값도 같이 본다. S는 무차원이라 불필요.
    if "오른어깨" in dyn:
        mid = lambda a, b: (dyn[a]["pos"] + dyn[b]["pos"]) / 2
        L = float(np.median(np.linalg.norm(mid("오른어깨", "왼어깨") - mid("오른엉덩이", "왼엉덩이"), axis=1)))
        out |= {"W_b": out["W"] / L**2, "W95_b": out["W95"] / L**2, "T_b": out["T"] / L, "F_b": out["F"] / L, "baseline_b": out["baseline"] / L}
    return out


def main() -> None:
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    results = []
    try:
        for motion, k, parts, best in RUNS:
            for preset in dict.fromkeys([best, "multi"]):
                joints = JOINT_PRESETS[preset]
                table = {m: [] for m in METRICS}
                for p in parts.split(","):
                    rows = [wst(load(f"data/raw/_nosound/{p}_S1_{motion}_T{t:02d}_CAM_F.mov", cache), joints)
                            for t in range(1, k + 1)]
                    for m in table:
                        table[m].append([r[m] for r in rows])
                icc = {m: icc_3_1(np.array(v)) for m, v in table.items()}
                # 축 간 중복 확인: 사람별 평균끼리의 상관
                pm = {m: np.array(v).mean(axis=1) for m, v in table.items()}
                corr = {f"{a}-{b}": float(np.corrcoef(pm[a], pm[b])[0, 1])
                        for a, b in [("W", "T"), ("W", "baseline"), ("T", "baseline"), ("S", "baseline")]}
                results.append({"motion": motion, "joints": preset, "n": len(parts.split(",")), "k": k,
                                "icc": icc, "corr": corr, "values": table})
                print(f"{motion} {preset:6s} n={len(parts.split(','))} k={k}  "
                      + "  ".join(f"{m}={v:+.3f}" for m, v in icc.items())
                      + "  | r " + " ".join(f"{c}={v:+.2f}" for c, v in corr.items()))
    finally:
        CACHE.write_bytes(pickle.dumps(cache))
    Path("reports/wst_pilot.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


def _z(values: dict, axes: list[str]) -> np.ndarray:
    """축별 z점수 (n, k, d). 에너지·가속·저크는 오른쪽 꼬리가 길어 log 후 표준화."""
    out = []
    for m in axes:
        a = np.array(values[m], float)
        a = a if m == "S" else np.log(a)
        out.append((a - a.mean()) / a.std())
    return np.stack(out, -1)


def identify(X: np.ndarray) -> float:
    """테이크 하나씩 빼고, 나머지 테이크 평균(사람별 중심)에 가장 가까운 사람을 맞힌 비율."""
    n, k, _ = X.shape
    hits = 0
    for t in range(k):
        cents = np.delete(X, t, axis=1).mean(axis=1)
        hits += sum(np.argmin(((cents - X[i, t]) ** 2).sum(1)) == i for i in range(n))
    return hits / (n * k)


def combine() -> None:
    """WST(+F)를 한 점수로 합친 ICC vs 4축 벡터로 본 식별률 비교 (reports/wst_pilot.json 기반)."""
    axes = ["W95_b", "T_b", "S", "F_b"]
    for r in json.loads(Path("reports/wst_pilot.json").read_text()):
        v = r["values"]
        if r["motion"] == "M03" and r["joints"] == "multi":  # P007 추적 실패 제외
            v = {m: x[:6] for m, x in v.items()}
        n = len(v["S"])
        line = f"{r['motion']} {r['joints']:6s} n={n} 우연={1/n:.2f} |"
        for name, ax in [("WST", axes[:3]), ("WSTF", axes)]:
            X = _z(v, ax)
            line += f" {name}: 합산ICC={icc_3_1(X.mean(-1)):+.3f} 식별={identify(X):.2f} |"
        line += " " + " ".join(f"{m}식별={identify(_z(v, [m])):.2f}" for m in axes)
        print(line)


def demo() -> None:
    """정의 자가검증: 직선 이동은 S≈1, 제자리 왕복은 S≈0, 빠를수록 W·T가 커진다."""
    t = np.linspace(0, 2, 60)
    line = np.stack([t, 0 * t], axis=1)
    osc = np.stack([np.sin(2 * np.pi * 4 * t), 0 * t], axis=1)
    assert directness(line, 15) > 0.99
    assert directness(osc, 15) < 0.3
    def rec(pos):
        sp = np.linalg.norm(np.gradient(pos, axis=0) * 30, axis=1)
        return {"dyn": {"x": {"pos": pos, "speed": sp + 1e-3}}, "fps": 30.0}  # 어깨 없음 → _b 생략
    slow, fast = wst(rec(osc * 0.5), ("x",)), wst(rec(osc), ("x",))
    assert fast["W"] > slow["W"] and fast["T"] > slow["T"] and fast["F"] > slow["F"]
    # 식별: 사람마다 뚜렷이 다른 값이면 100% 맞혀야 한다
    X = np.array([[[0.0], [0.1]], [[5.0], [5.1]], [[10.0], [9.9]]])
    assert identify(X) == 1.0
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--combine" in sys.argv:
        combine()
    else:
        main()
