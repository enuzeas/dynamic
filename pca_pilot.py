"""트랙 1 — 포즈 공간 PCA 파일럿 (n=13, P001–P014).

수업 슬라이드의 "x = P·z + x_m"(포즈를 저차원 z로 압축)을 식별에 써본다:
  1. 같은 동작의 모든 사람·테이크 프레임을 모아 PCA → 주성분(PC) 축
  2. 테이크마다 PC 점수 z(t)의 평균(자세)·표준편차(움직임 폭)·PC1 주기(케이던스)를 특징으로
  3. 특징별 ICC(3,1)과 프로필 벡터 식별률을 기존 최고 지표(W95_b, WSTF)와 비교
  4. (Chai 2005 아이디어) 상위 K개 PC로 복원이 안 되는 프레임 = 추적 실패 후보

정규화: 골반 중심 빼고 몸통 길이로 나눔 — 체격·촬영 거리 착시 제거(track1_wst_pilot.md §1-5).
기준선(W95_b 등)은 wst_pilot.wst()를 같은 랜드마크로 다시 계산 — P001–P007 값이 기존 파일럿과 같아야 함.

사용법 (repo 루트, mediapipe 설치된 venv):
  .venv/bin/python pca_pilot.py            # 결과 표 + reports/pca_pilot.json
  .venv/bin/python pca_pilot.py --demo     # 자가검증
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import _make_landmarker, _smooth, LANDMARK_INDEX  # noqa: E402
from build_icc_features import _trim_window  # noqa: E402
from evaluate_icc import icc_3_1  # noqa: E402
from wst_pilot import wst, identify, _z  # noqa: E402

RAW = Path("data/raw/_nosound")
CACHE = Path("reports/pca_pilot_cache.pkl")  # ponytail: 영상당 MediaPipe 1회만, 지우면 재추출
BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 어깨·팔꿈치·손목·골반·무릎·발목
N_PC = 10
# (동작, 테이크 수, 참가자) — M06은 P005 제외(discarded_takes), P009 제외(첫 실행에서 복원 오차 검사가
# 잡아냄: 조깅 중 화면 밖으로 나가고 카메라 바로 앞에서 다리가 잘림 → W95_b가 다른 사람의 ~200배),
# M04는 P010·P011 미촬영
RUNS = [
    ("M06", 2, "P001,P002,P003,P004,P006,P007,P008,P010,P011,P012,P013,P014", "hips"),
    ("M04", 3, "P001,P002,P003,P004,P005,P006,P007,P008,P009,P012,P013,P014", "hips"),
]


def extract(path: Path) -> dict:
    """영상 → 33관절 (x,y) 프레임 대각선 정규화 좌표 + fps. 검출 실패 프레임은 직전 값 유지(pose_extract와 동일)."""
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    diag = (w ** 2 + h ** 2) ** 0.5
    lm = _make_landmarker()
    out, last, i = [], np.zeros((33, 2)), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                    int(i * 1000 / fps))
            if r.pose_landmarks:
                last = np.array([(p.x * w / diag, p.y * h / diag) for p in r.pose_landmarks[0]])
            out.append(last)
            i += 1
    finally:
        cap.release()
        lm.close()
    return {"xy": np.array(out), "fps": fps}


def dyn8(rec: dict) -> dict:
    """extract() 결과 → wst_pilot.wst()가 받는 형식(pose_extract.extract_joint_dynamics와 같은 계산)."""
    fps, dyn = rec["fps"], {}
    for name, j in LANDMARK_INDEX.items():
        pos = _smooth(rec["xy"][:, j].astype(float))
        speed = _smooth(np.linalg.norm(np.gradient(pos, axis=0) * fps, axis=1))
        dyn[name] = {"pos": pos, "speed": speed}
    return {"dyn": dyn, "fps": fps}


def body_frames(rec: dict) -> np.ndarray:
    """트림 구간의 정규화 포즈 (T, 24): 골반 중심 기준, 몸통 길이로 나눔."""
    d = dyn8(rec)["dyn"]
    win = _trim_window(np.mean([d[j]["speed"] for j in ("오른엉덩이", "왼엉덩이")], axis=0))
    xy = np.stack([_smooth(rec["xy"][:, j].astype(float)) for j in range(33)], axis=1)[win]
    hip = (xy[:, 23] + xy[:, 24]) / 2
    sh = (xy[:, 11] + xy[:, 12]) / 2
    L = np.median(np.linalg.norm(sh - hip, axis=1))
    return ((xy[:, BODY] - hip[:, None]) / L).reshape(len(xy), -1)


def pca(frames: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = frames.mean(axis=0)
    _, s, vt = np.linalg.svd(frames - mean, full_matrices=False)
    return mean, vt[:k], (s ** 2 / (s ** 2).sum())[:k]


def take_features(z: np.ndarray, fps: float) -> dict[str, float]:
    f = {}
    for c in range(z.shape[1]):
        f[f"PC{c + 1}_mean"] = float(z[:, c].mean())
        f[f"PC{c + 1}_std"] = float(z[:, c].std())
    # PC1 주기(케이던스): 0.5–4Hz에서 가장 센 주파수
    x = z[:, 0] - z[:, 0].mean()
    freqs = np.fft.rfftfreq(len(x), 1 / fps)
    band = (freqs > 0.5) & (freqs < 4.0)
    f["PC1_cadence"] = float(freqs[band][np.argmax(np.abs(np.fft.rfft(x))[band])]) if band.any() else float("nan")
    return f


def main() -> None:
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    report = []
    try:
        for motion, k, parts, joints in RUNS:
            people = parts.split(",")
            recs = {}
            for p in people:
                for t in range(1, k + 1):
                    path = RAW / f"{p}_S1_{motion}_T{t:02d}_CAM_F.mov"
                    if str(path) not in cache:
                        cache[str(path)] = extract(path)
                    recs[(p, t)] = cache[str(path)]
            frames = {key: body_frames(r) for key, r in recs.items()}
            mean, comps, evr = pca(np.concatenate([f[::2] for f in frames.values()]), N_PC)
            feats = {key: take_features((f - mean) @ comps.T, recs[key]["fps"]) for key, f in frames.items()}
            # 기준선: 기존 파일럿과 같은 wst() (엉덩이 관절)
            from build_icc_features import JOINT_PRESETS
            base = {key: wst(dyn8(r), JOINT_PRESETS[joints]) for key, r in recs.items()}

            table = lambda src, m: [[src[(p, t)][m] for t in range(1, k + 1)] for p in people]
            names = list(next(iter(feats.values())))
            icc = {m: icc_3_1(np.array(table(feats, m))) for m in names}
            icc_base = {m: icc_3_1(np.array(table(base, m))) for m in ("W95_b", "T_b", "S", "F_b", "baseline_b")}
            vals = {**{m: table(feats, m) for m in names}, **{m: table(base, m) for m in icc_base}}

            def ident(ms, log=False):
                cols = []
                for m in ms:
                    a = np.array(vals[m], float)
                    a = np.log(a) if log and (a > 0).all() else a
                    cols.append((a - a.mean()) / a.std())
                return identify(np.stack(cols, -1))
            std5 = [f"PC{c}_std" for c in range(1, 6)]
            all20 = [f"PC{c}_{s}" for c in range(1, N_PC + 1) for s in ("mean", "std")]
            idr = {
                "W95_b 단독": identify(_z({m: vals[m] for m in ["W95_b"]}, ["W95_b"])),
                "WSTF 벡터": identify(_z({m: vals[m] for m in ["W95_b", "T_b", "S", "F_b"]}, ["W95_b", "T_b", "S", "F_b"])),
                "PCA 움직임폭(PC1–5 std)": ident(std5),
                "PCA 자세+폭(PC1–10 mean·std)": ident(all20),
                "PCA 자세+폭 + WSTF": identify(np.concatenate([
                    np.stack([(np.array(vals[m]) - np.mean(vals[m])) / np.std(vals[m]) for m in all20], -1),
                    _z({m: vals[m] for m in ["W95_b", "T_b", "S", "F_b"]}, ["W95_b", "T_b", "S", "F_b"])], -1)),
            }

            # 추적 실패 후보: PC 10개로 복원한 오차가 큰 프레임 비율
            resid = {}
            for key, f in frames.items():
                c = f - mean
                err = np.linalg.norm(c - (c @ comps.T) @ comps, axis=1)
                resid[key] = err
            allerr = np.concatenate(list(resid.values()))
            thr = float(np.percentile(allerr, 99))
            worst = sorted(((float((e > thr).mean()), f"{p}_T{t:02d}") for (p, t), e in resid.items()), reverse=True)[:5]

            print(f"\n=== {motion}  n={len(people)} k={k}  (우연 식별률 {1 / len(people):.0%})")
            print("설명 분산 PC1–10:", " ".join(f"{v:.0%}" for v in evr), f"(누적 {evr.sum():.0%})")
            print("기준선 ICC:", "  ".join(f"{m}={v:+.3f}" for m, v in icc_base.items()))
            top = sorted(icc.items(), key=lambda kv: -kv[1])[:8]
            print("PCA 특징 ICC 상위:", "  ".join(f"{m}={v:+.3f}" for m, v in top))
            print("식별률:", "  ".join(f"{m}={v:.0%}" for m, v in idr.items()))
            print("복원 오차 상위 1% 프레임이 몰린 테이크:", worst)
            report.append({"motion": motion, "n": len(people), "k": k, "evr": evr.tolist(), "icc_base": icc_base,
                           "icc_pca": icc, "ident": idr, "resid_worst": worst})
    finally:
        CACHE.write_bytes(pickle.dumps(cache))
    Path("reports/pca_pilot.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


def demo() -> None:
    """PCA 자가검증: 2차원 안에 놓인 24차원 데이터는 PC 2개로 분산 거의 100%, 복원 오차 ~0."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(24, 2)))[0].T
    x = rng.normal(size=(500, 2)) * [3, 1] @ basis + 5
    mean, comps, evr = pca(x, 3)
    assert evr[:2].sum() > 0.999 and np.allclose((x - mean) - ((x - mean) @ comps[:2].T) @ comps[:2], 0, atol=1e-9)
    # 케이던스: 2Hz 사인파
    t = np.arange(300) / 60
    assert abs(take_features(np.sin(2 * np.pi * 2 * t)[:, None], 60)["PC1_cadence"] - 2.0) < 0.25
    print("demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
