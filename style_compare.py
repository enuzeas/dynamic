"""스타일 전이 비교 — 참가자 조깅 스타일을 다른 동작(CMU 걷기)에 입히는 두 방법.

  A. Motion Puzzle (학습된 latent 스타일 전이): content=CMU 걷기 35_06, style=참가자 조깅(3D)
  C·D. Aberman et al. 2020 (deep-motion-editing): style=참가자 3D BVH(C) 또는 영상 2D 관절(D)
  B. 관절 각도 오프셋 (방법 3): CMU 걷기에 참가자의 "평균 대비" 자세 습관(팔꿈치·무릎·팔 벌림·상체)과
     팔 흔들림 폭 비율을 직접 적용 — 학습 없음, 무엇을 바꿨는지 정확히 앎

참가자 3D는 MediaPipe world landmarks(mp3d_extract.py)로 근사 — GVHMR(GPU)보다 거칠다.
3D 관절 위치 → CMU 31관절 BVH는 뼈 방향을 맞추는 IK(ik())로 만든다. CMU 골격 비율에 얹으므로 체형은 빠진다.

평가 (같은 3D 지표로):
  ① 옮겨졌나: 참가자 조깅 지표 vs 결과 걷기 지표의 사람 간 상관
  ② 진짜 걷기를 닮았나: 결과 걷기 vs 그 사람의 실제 걷기(M01)
  ③ 차이가 남았나: 결과들 사이 사람 간 퍼짐 / 실제 걷기의 퍼짐
  ④ 누구인지 맞히기: 결과 걷기 → 실제 걷기 / 원천 조깅 최근접

motion_puzzle conda env에서 실행 (Animation/BVH 모듈이 옛 numpy 필요):
  conda run --no-capture-output -n motion_puzzle python style_compare.py           # 조깅 → 걷기
  conda run --no-capture-output -n motion_puzzle python style_compare.py --same    # 걷기 → 걷기(안 쓴 테이크와 비교)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "external" / "motion_puzzle" / "motion"))
sys.path.insert(0, str(ROOT))
import Animation as AnimModule  # noqa: E402
from Animation import Animation  # noqa: E402
from Quaternions import Quaternions  # noqa: E402
import BVH  # noqa: E402
import retarget_smpl_to_cmu as R  # noqa: E402

# POSE_SRC=gvhmr 이면 참가자 3D를 GVHMR/FootMR(gvhmr_to_npz.py)에서 읽고 결과도 따로 둔다
SRC = os.environ.get("POSE_SRC", "mp3d")
MP3D = ROOT / "reports" / SRC
OUT = ROOT / "reports" / ("style_compare" if SRC == "mp3d" else f"style_compare_{SRC}")
MPZ = ROOT / "external" / "motion_puzzle"
CONTENT = {"walk": MPZ / "datasets" / "cmu" / "test_bvh" / "35_06.bvh"}
PEOPLE = "P001,P002,P003,P004,P006,P007,P008,P010,P011,P012,P013,P014".split(",")
N31 = R.RAW31_NAMES
PAR = R.RAW31_PARENTS
OFF = R.RAW31_OFFSETS
J = {n: i for i, n in enumerate(N31)}
UNITS = R.UNITS_PER_M


# ---------------------------------------------------------------- 회전 도구 (행렬, 열벡터)
def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def between(a, b):
    """a→b 최소 회전 (…,3,3). a, b: (…,3)"""
    a, b = nrm(a), nrm(b)
    v = np.cross(a, b)
    c = (a * b).sum(-1)[..., None, None]
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + K + K @ K / (1 + c + 1e-12)


def frame(up, across):
    y = nrm(up)
    x = nrm(across - (across * y).sum(-1, keepdims=True) * y)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def rot_axis(axis, ang):
    """축 axis(…,3)를 중심으로 ang(…) 라디안 회전 행렬"""
    k = nrm(axis)
    K = np.zeros(k.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    return np.broadcast_to(np.eye(3), K.shape) + s * K + (1 - c) * K @ K


def mat2quat(M):
    w = np.sqrt(np.maximum(0, 1 + M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2])) / 2
    x = np.sqrt(np.maximum(0, 1 + M[..., 0, 0] - M[..., 1, 1] - M[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - M[..., 0, 0] + M[..., 1, 1] - M[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - M[..., 0, 0] - M[..., 1, 1] + M[..., 2, 2])) / 2
    x = np.copysign(x, M[..., 2, 1] - M[..., 1, 2])
    y = np.copysign(y, M[..., 0, 2] - M[..., 2, 0])
    z = np.copysign(z, M[..., 1, 0] - M[..., 0, 1])
    return np.stack([w, x, y, z], -1)


REST = np.zeros((31, 3))
for j in range(1, 31):
    REST[j] = REST[PAR[j]] + OFF[j]


# ---------------------------------------------------------------- 3D 관절 위치 → CMU 31관절 BVH
KEYS = ["pelvis", "Lhip", "Rhip", "Lknee", "Rknee", "Lank", "Rank", "Ltoe", "Rtoe", "Lsh", "Rsh", "Lel", "Rel",
        "Lwr", "Rwr", "Lidx", "Ridx", "neck", "head"]
LIMBS = [  # (회전하는 관절, 그 뼈의 쉴때 끝 관절, 목표 시작점, 목표 끝점)
    ("LeftUpLeg", "LeftLeg", "Lhip", "Lknee"), ("LeftLeg", "LeftFoot", "Lknee", "Lank"), ("LeftFoot", "LeftToeBase", "Lank", "Ltoe"),
    ("RightUpLeg", "RightLeg", "Rhip", "Rknee"), ("RightLeg", "RightFoot", "Rknee", "Rank"), ("RightFoot", "RightToeBase", "Rank", "Rtoe"),
    ("Neck1", "Head", "neck", "head"),
    ("LeftArm", "LeftForeArm", "Lsh", "Lel"), ("LeftForeArm", "LeftHand", "Lel", "Lwr"), ("LeftHand", "LeftHandIndex1", "Lwr", "Lidx"),
    ("RightArm", "RightForeArm", "Rsh", "Rel"), ("RightForeArm", "RightHand", "Rel", "Rwr"), ("RightHand", "RightHandIndex1", "Rwr", "Ridx"),
]


def ik(pts: dict, root_units: np.ndarray) -> Animation:
    """pts: 이름→(T,3) CMU 축(Y 위, 사람 앞 +Z, 사람 왼쪽 +X) 관절 위치(단위 무관 — 방향만 씀)."""
    T = len(root_units)
    G = np.tile(np.eye(3), (T, 31, 1, 1))
    rest_up = (REST[J["LeftArm"]] + REST[J["RightArm"]]) / 2 - REST[0]  # 목표의 "골반→어깨 중점"과 같은 정의
    hips_r = frame(rest_up, REST[J["LeftUpLeg"]] - REST[J["RightUpLeg"]])
    hips_t = frame(pts["neck"] - pts["pelvis"], pts["Lhip"] - pts["Rhip"])
    G[:, 0] = hips_t @ hips_r.T
    chest_r = frame(rest_up, REST[J["LeftArm"]] - REST[J["RightArm"]])
    chest_t = frame(pts["neck"] - pts["pelvis"], pts["Lsh"] - pts["Rsh"])
    done = {0}
    for j in range(1, 31):
        G[:, j] = G[:, PAR[j]]  # 기본: 로컬 항등
        if N31[j] == "Spine1":
            G[:, j] = chest_t @ chest_r.T
        for rot, end, a, b in LIMBS:
            if N31[j] == rot:
                rest_dir = REST[J[end]] - REST[j]
                v = (G[:, PAR[j]] @ rest_dir[:, None])[..., 0]
                G[:, j] = between(v, pts[b] - pts[a]) @ G[:, PAR[j]]
        done.add(j)
    L = np.empty_like(G)
    L[:, 0] = G[:, 0]
    for j in range(1, 31):
        L[:, j] = np.swapaxes(G[:, PAR[j]], -1, -2) @ G[:, j]
    rot = Quaternions(mat2quat(L))
    pos = np.tile(OFF[None], (T, 1, 1))
    pos[:, 0] = root_units
    anim = Animation(rot, pos, Quaternions.id(31), OFF.copy(), PAR.copy())
    return R.ground(anim)


def mp_points(npz: Path, d0: float = 5.0):
    """MediaPipe world(또는 GVHMR 관절) → CMU 축 관절 위치(미터) + 루트 경로(미터)."""
    d = np.load(npz)
    if "joints" in d:
        return gvhmr_points(d)
    W = d["world"].copy()
    W[..., 1] *= -1
    W[..., 2] *= -1
    win = min(9, len(W) // 2 * 2 - 1)
    W = savgol_filter(W, win, 2, axis=0)
    mid = lambda a, b: (W[:, a] + W[:, b]) / 2
    idx = {"Lhip": 23, "Rhip": 24, "Lknee": 25, "Rknee": 26, "Lank": 27, "Rank": 28, "Ltoe": 31, "Rtoe": 32,
           "Lsh": 11, "Rsh": 12, "Lel": 13, "Rel": 14, "Lwr": 15, "Rwr": 16, "Lidx": 19, "Ridx": 20}
    pts = {k: W[:, v] for k, v in idx.items()}
    pts["pelvis"], pts["neck"], pts["head"] = mid(23, 24), mid(11, 12), mid(7, 8)
    xy = d["xy"]
    hip = (xy[:, 23] + xy[:, 24]) / 2
    tor = savgol_filter(np.linalg.norm((xy[:, 11] + xy[:, 12]) / 2 - hip, axis=1), min(31, len(xy) // 2 * 2 - 1), 2)
    m_per = 0.5 / tor  # 몸통 ≈ 0.5 m
    root = np.stack([(hip[:, 0] - hip[0, 0]) * m_per, -(hip[:, 1] - hip[:, 1].mean()) * m_per,
                     d0 * (1 - np.median(tor) / tor)], axis=1)
    root = savgol_filter(root, min(15, len(root) // 2 * 2 - 1), 2, axis=0)
    return pts, root, float(d["fps"])


def gvhmr_points(d):
    """gvhmr_to_npz.py 결과(SMPL-X 22관절, 월드 Y 위) → mp_points와 같은 꼴. 골반 중심은 MediaPipe처럼 두 엉덩이 중점."""
    P = d["joints"]
    pel = (P[:, 1] + P[:, 2]) / 2
    idx = {"Lhip": 1, "Rhip": 2, "Lknee": 4, "Rknee": 5, "Lank": 7, "Rank": 8, "Ltoe": 10, "Rtoe": 11,
           "Lsh": 16, "Rsh": 17, "Lel": 18, "Rel": 19, "Lwr": 20, "Rwr": 21, "head": 15}
    pts = {k: P[:, v] - pel for k, v in idx.items()}
    pts["pelvis"], pts["neck"] = np.zeros_like(pel), (pts["Lsh"] + pts["Rsh"]) / 2
    for s in "LR":  # ponytail: SMPL-X 몸 파라미터엔 손가락이 없어 손은 아래팔을 곧게 이은 것으로 둔다
        pts[s + "idx"] = pts[s + "wr"] + 0.4 * (pts[s + "wr"] - pts[s + "el"])
    root = pel - np.array([pel[0, 0], pel[:, 1].mean(), pel[0, 2]])
    return pts, root, float(d["fps"])


def bvh_points(anim: Animation, names: list[str]) -> dict:
    P = AnimModule.positions_global(anim)
    ix = {n: i for i, n in enumerate(names)}
    g = lambda n: P[:, ix[n]]
    pts = {"pelvis": g("Hips"), "Lhip": g("LeftUpLeg"), "Rhip": g("RightUpLeg"), "Lknee": g("LeftLeg"), "Rknee": g("RightLeg"),
           "Lank": g("LeftFoot"), "Rank": g("RightFoot"), "Ltoe": g("LeftToeBase"), "Rtoe": g("RightToeBase"),
           "Lsh": g("LeftArm"), "Rsh": g("RightArm"), "Lel": g("LeftForeArm"), "Rel": g("RightForeArm"),
           "Lwr": g("LeftHand"), "Rwr": g("RightHand"), "head": g("Head"),
           "Lidx": g("LeftHandIndex1") if "LeftHandIndex1" in ix else g("LeftHand"),
           "Ridx": g("RightHandIndex1") if "RightHandIndex1" in ix else g("RightHand")}
    pts["neck"] = (pts["Lsh"] + pts["Rsh"]) / 2
    return pts


def save(anim: Animation, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    BVH.save(str(path), anim, names=N31, frametime=1.0 / fps)


# ---------------------------------------------------------------- 3D 지표 (각도, 체형 무관)
METRICS = ("팔꿈치", "무릎", "팔 벌림", "상체 숙임", "팔 흔들림", "허벅지 흔들림")


def metrics(pts: dict) -> dict[str, float]:
    def ang(u, v):
        return np.degrees(np.arccos(np.clip((nrm(u) * nrm(v)).sum(-1), -1, 1)))
    up = nrm(pts["neck"] - pts["pelvis"])
    across = nrm(pts["Lhip"] - pts["Rhip"])
    fwd = nrm(np.cross(across, up))
    down = -up

    def sag(vec):  # 몸 기준 앞뒤 흔들림 각 (앞 +)
        return np.degrees(np.arctan2((vec * fwd).sum(-1), (vec * down).sum(-1)))
    rng = lambda a: float(np.percentile(a, 95) - np.percentile(a, 5))
    vert = np.array([0, 1.0, 0])
    lean = np.degrees(np.arctan2((up * np.cross(across, vert)).sum(-1), (up * vert).sum(-1)))
    return {
        "팔꿈치": float(np.mean([ang(pts[s] - pts[e], pts[w] - pts[e]).mean() for s, e, w in (("Lsh", "Lel", "Lwr"), ("Rsh", "Rel", "Rwr"))])),
        "무릎": float(np.mean([ang(pts[h] - pts[k], pts[a] - pts[k]).mean() for h, k, a in (("Lhip", "Lknee", "Lank"), ("Rhip", "Rknee", "Rank"))])),
        "팔 벌림": float(np.mean([ang(down, pts[e] - pts[s]).mean() for s, e in (("Lsh", "Lel"), ("Rsh", "Rel"))])),
        "상체 숙임": float(lean.mean()),
        "팔 흔들림": float(np.mean([rng(sag(pts[e] - pts[s])) for s, e in (("Lsh", "Lel"), ("Rsh", "Rel"))])),
        "허벅지 흔들림": float(np.mean([rng(sag(pts[k] - pts[h])) for h, k in (("Lhip", "Lknee"), ("Rhip", "Rknee"))])),
    }


# ---------------------------------------------------------------- 방법 B: 관절 각도 오프셋
def apply_offsets(pts: dict, d: dict, swing_ratio: float) -> dict:
    """d: 평균 대비 각도 차(도). 팔꿈치·무릎은 펴는 쪽 +, 팔 벌림은 바깥 +, 상체는 앞 +. swing_ratio: 팔 흔들림 폭 배율."""
    p = {k: v.copy() for k, v in pts.items()}
    rad = np.radians
    up = nrm(p["neck"] - p["pelvis"])
    across = nrm(p["Lhip"] - p["Rhip"])
    fwd = nrm(np.cross(across, up))

    def rotate(names, center, Rm):
        for n in names:
            p[n] = center + (Rm @ (p[n] - center)[..., None])[..., 0]
    # 상체 숙임: 골반을 중심으로 좌우 축 회전
    Rl = rot_axis(across, np.full(len(up), rad(d["상체 숙임"])))
    rotate(["Lsh", "Rsh", "Lel", "Rel", "Lwr", "Rwr", "Lidx", "Ridx", "neck", "head"], p["pelvis"], Rl)
    up = nrm(p["neck"] - p["pelvis"])
    fwd = nrm(np.cross(across, up))
    for side, sgn in (("L", 1), ("R", -1)):
        sh, el, wr, ix = p[side + "sh"], side + "el", side + "wr", side + "idx"
        # 팔 흔들림 폭: 앞뒤 각을 평균 둘레로 배율
        arm = p[el] - sh
        a = np.arctan2((arm * fwd).sum(-1), (arm * -up).sum(-1))
        rotate([el, wr, ix], sh, rot_axis(np.cross(-up, fwd), (a - a.mean()) * (swing_ratio - 1)))
        # 팔 벌림: 앞 방향 축으로 바깥쪽
        rotate([el, wr, ix], sh, rot_axis(fwd, np.full(len(up), sgn * rad(d["팔 벌림"]))))
        # 팔꿈치: 팔 평면 법선 축으로 펴기
        n = np.cross(sh - p[el], p[wr] - p[el])
        rotate([wr, ix], p[el], rot_axis(n, np.full(len(up), rad(d["팔꿈치"]))))
    for side in ("L", "R"):
        hp, kn, an, to = p[side + "hip"], side + "knee", side + "ank", side + "toe"
        n = np.cross(hp - p[kn], p[an] - p[kn])
        rotate([an, to], p[kn], rot_axis(n, np.full(len(up), rad(d["무릎"]))))
    return p


# ---------------------------------------------------------------- 방법 E: 관절 각도 공간 AdaIN
# AdaIN(Huang & Belongie 2017; 동작에서는 MOCHA·Aberman 등이 학습된 특징에 씀)을 사람이 읽는 관절 각도 채널에 직접 적용:
#   c'(t) = μ_s + (σ_s / σ_c) · (c(t) − μ_c)      — 콘텐츠의 평균·흔들림을 지우고 스타일 것을 입힘
# "상대" 판은 μ_s, σ_s 대신 그 사람이 집단 평균에서 벗어난 만큼만 옮긴다(측정 방식 차이·동작 차이 상쇄):
#   μ' = μ_c + (μ_s − μ_집단),  σ' = σ_c · σ_s / σ_집단
# 채널(15개): 상체·고개의 앞뒤/옆 기울기, 위팔·허벅지의 앞뒤/옆 각(몸통 좌표계), 팔꿈치·무릎 굽힘. 발·손은 아래 뼈와 같이 돈다.
CHANNELS = ["trunk_sag", "trunk_front", "head_sag", "head_front",
            "Larm_sag", "Larm_front", "Rarm_sag", "Rarm_front", "Lelbow", "Relbow",
            "Lthigh_sag", "Lthigh_front", "Rthigh_sag", "Rthigh_front", "Lknee", "Rknee"]


def _body_frame(pts):
    across = nrm(pts["Lhip"] - pts["Rhip"])
    vert = np.broadcast_to(np.array([0, 1.0, 0]), across.shape)
    fwd = nrm(np.cross(across, vert))
    return across, vert, fwd


def _basis(down, fwd, out):
    """down·fwd·out을 정규직교로(그람-슈미트). 몸통·골반 좌우선이 기울면 원래 축들이 정확히 직교하지 않는다."""
    d = nrm(down)
    f = nrm(fwd - (fwd * d).sum(-1, keepdims=True) * d)
    o = np.cross(d, f)
    o = o * np.sign((o * out).sum(-1, keepdims=True) + 1e-12)
    return d, f, o


def _two_angles(v, down, fwd, out):
    """뼈 방향 v를 (앞뒤 각, 바깥 각)으로 — down 기준. 도."""
    down, fwd, out = _basis(down, fwd, out)
    return (np.degrees(np.arctan2((v * fwd).sum(-1), (v * down).sum(-1))),
            np.degrees(np.arctan2((v * out).sum(-1), (v * down).sum(-1))))


def _from_two_angles(a, b, down, fwd, out):
    down, fwd, out = _basis(down, fwd, out)
    a, b = np.radians(np.clip(a, -80, 80)), np.radians(np.clip(b, -80, 80))
    return nrm(down + np.tan(a)[:, None] * fwd + np.tan(b)[:, None] * out)


def _flex(p, a, b, c):
    u, v = p[a] - p[b], p[c] - p[b]
    return np.degrees(np.arccos(np.clip((nrm(u) * nrm(v)).sum(-1), -1, 1)))


def channels(pts: dict) -> dict:
    across, vert, fwd = _body_frame(pts)
    up = nrm(pts["neck"] - pts["pelvis"])
    ch = {}
    ch["trunk_sag"], ch["trunk_front"] = _two_angles(up, vert, fwd, -across)  # 위쪽 기준: 앞으로 숙임 +, 오른쪽 기움 +
    tdown = -up
    tacross = nrm(pts["Lsh"] - pts["Rsh"])
    tfwd = nrm(np.cross(tacross, up))
    ch["head_sag"], ch["head_front"] = _two_angles(nrm(pts["head"] - pts["neck"]), up, tfwd, -tacross)
    for s, out in (("L", tacross), ("R", -tacross)):
        ch[s + "arm_sag"], ch[s + "arm_front"] = _two_angles(nrm(pts[s + "el"] - pts[s + "sh"]), tdown, tfwd, out)
        ch[s + "elbow"] = _flex(pts, s + "sh", s + "el", s + "wr")
    for s, out in (("L", across), ("R", -across)):
        ch[s + "thigh_sag"], ch[s + "thigh_front"] = _two_angles(nrm(pts[s + "knee"] - pts[s + "hip"]), -vert, fwd, out)
        ch[s + "knee"] = _flex(pts, s + "hip", s + "knee", s + "ank")
    return ch


def stats(ch: dict) -> dict:
    return {k: (float(np.mean(v)), float(np.std(v) + 1e-6)) for k, v in ch.items()}


def rebuild(pts: dict, new: dict) -> dict:
    """콘텐츠 관절 위치 pts에서 각도 채널을 new 값으로 바꾼 관절 위치. 뼈 길이는 그대로."""
    p = {k: v.copy() for k, v in pts.items()}
    L = lambda a, b: np.linalg.norm(pts[b] - pts[a], axis=-1, keepdims=True)
    across, vert, fwd = _body_frame(pts)
    # 상체: 골반 기준으로 몸통 방향을 새 각도로 돌리고, 위쪽 관절을 통째로 같이 돌림
    old_up = nrm(pts["neck"] - pts["pelvis"])
    new_up = _from_two_angles(new["trunk_sag"], new["trunk_front"], vert, fwd, -across)
    Rt = between(old_up, new_up)
    for k in ("neck", "head", "Lsh", "Rsh", "Lel", "Rel", "Lwr", "Rwr", "Lidx", "Ridx"):
        p[k] = p["pelvis"] + (Rt @ (pts[k] - pts["pelvis"])[..., None])[..., 0]
    up = nrm(p["neck"] - p["pelvis"])
    tacross = nrm(p["Lsh"] - p["Rsh"])
    tfwd = nrm(np.cross(tacross, up))
    p["head"] = p["neck"] + _from_two_angles(new["head_sag"], new["head_front"], up, tfwd, -tacross) * L("neck", "head")

    def chain(root, mid, end, tip, sag, front, flex, down, fw, out):
        old_seg = p[mid] - p[root]
        seg = _from_two_angles(sag, front, down, fw, out) * L(root, mid)
        Rs = between(old_seg, seg)
        lower = (Rs @ (p[end] - p[mid])[..., None])[..., 0]
        tipv = (Rs @ (p[tip] - p[end])[..., None])[..., 0]
        p[mid] = p[root] + seg
        # 굽힘: 위·아래 뼈가 이루는 평면에서 아래 뼈를 돌려 목표 각도로
        cur = np.degrees(np.arccos(np.clip((nrm(-seg) * nrm(lower)).sum(-1), -1, 1)))
        n = np.cross(-seg, lower)
        Rf = rot_axis(n, np.radians(np.clip(flex, 30, 179.5) - cur))
        lower = (Rf @ lower[..., None])[..., 0]
        tipv = (Rf @ tipv[..., None])[..., 0]
        p[end] = p[mid] + lower
        p[tip] = p[end] + tipv
    for s, out in (("L", tacross), ("R", -tacross)):
        chain(s + "sh", s + "el", s + "wr", s + "idx", new[s + "arm_sag"], new[s + "arm_front"], new[s + "elbow"], -up, tfwd, out)
    for s, out in (("L", across), ("R", -across)):
        chain(s + "hip", s + "knee", s + "ank", s + "toe", new[s + "thigh_sag"], new[s + "thigh_front"], new[s + "knee"], -vert, fwd, out)
    return p


def adain(content_pts: dict, style: dict, group: dict | None = None, frontal_sd: bool = False) -> dict:
    """style·group: stats() 결과. group이 있으면 '상대' AdaIN(집단 평균 대비 차이만 옮김).
    frontal_sd=False(기본): 옆 방향(*_front) 채널은 평균만 옮기고 흔들림 폭은 콘텐츠 것을 둔다 — 참가자가 카메라 앞을
    가로질러 걸어서 몸의 옆 방향이 카메라 깊이 방향이 되고, MediaPipe 깊이 추정 노이즈가 흔들림 폭으로 들어오기 때문
    (실측: 상체 옆 흔들림 기준 걷기 1.0° → 그대로 옮기면 8.0°)."""
    ch = channels(content_pts)
    cs = stats(ch)
    new = {}
    for k, c in ch.items():
        mc, sc = cs[k]
        ms, ss = style[k]
        if group is not None:
            mg, sg = group[k]
            ms, ss = mc + (ms - mg), sc * ss / sg
        if k.endswith("_front") and not frontal_sd:
            ss = sc
        new[k] = ms + (ss / sc) * (c - mc)
    return rebuild(content_pts, new)


def group_stats(all_stats: list[dict]) -> dict:
    return {k: (float(np.mean([s[k][0] for s in all_stats])), float(np.mean([s[k][1] for s in all_stats]))) for k in all_stats[0]}


# ---------------------------------------------------------------- 방법 C·D: Aberman et al. 2020 (deep-motion-editing)
# "Unpaired Motion Style Transfer from Video to Animation" — 같은 CMU 31관절 골격. 스타일을 3D BVH로도,
# 영상의 2D 관절(OpenPose JSON)로도 받는다 → 2D는 3D 추정을 거치지 않으므로 MediaPipe 2D를 그대로 넣는다.
DME = ROOT / "external" / "deep-motion-editing"
TREADMILL = DME / "style_transfer" / "data" / "treadmill" / "json_inputs" / "27"
BODY25_FROM_MP = {0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 9: 24, 10: 26, 11: 28, 12: 23, 13: 25, 14: 27,
                  15: 5, 16: 2, 17: 8, 18: 7, 19: 31, 21: 29, 22: 32, 24: 30}


def _treadmill_torso() -> float:
    ts = []
    for f in sorted(TREADMILL.glob("*.json"))[:60]:
        k = np.array(json.loads(f.read_text())["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
        ts.append(np.linalg.norm(k[1, :2] - k[8, :2]))
    return float(np.median(ts))


def openpose_dir(npz: Path, dst: Path) -> Path:
    """MediaPipe 2D(프레임 대각선 정규화) → OpenPose BODY_25 JSON 폴더. 학습 데이터(트레드밀 영상)와 몸통 크기를 맞추고 30fps로."""
    d = np.load(ROOT / "reports" / "mp3d" / npz.name)  # Aberman 2D 입력은 언제나 MediaPipe 2D
    xy = d["xy"] * np.hypot(1920, 1080)
    step = max(1, int(round(float(d["fps"]) / 30)))
    xy = xy[::step]
    neck, mid = (xy[:, 11] + xy[:, 12]) / 2, (xy[:, 23] + xy[:, 24]) / 2
    xy = xy * (_treadmill_torso() / np.median(np.linalg.norm(neck - mid, axis=1)))
    neck, mid = (xy[:, 11] + xy[:, 12]) / 2, (xy[:, 23] + xy[:, 24]) / 2
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.json"):
        f.unlink()
    for t in range(len(xy)):
        k = np.zeros((25, 3))
        for b, m in BODY25_FROM_MP.items():
            k[b] = [*xy[t, m], 0.9]
        k[1], k[8] = [*neck[t], 0.9], [*mid[t], 0.9]
        k[20], k[23] = k[19], k[22]
        lh, rh = np.zeros((21, 3)), np.zeros((21, 3))
        lh[10], rh[10] = [*xy[t, 19], 0.9], [*xy[t, 20], 0.9]  # 손끝 근사(검지)
        (dst / f"{t:04d}.json").write_text(json.dumps({"version": 1.3, "people": [{
            "person_id": [-1], "pose_keypoints_2d": k.ravel().tolist(), "face_keypoints_2d": [],
            "hand_left_keypoints_2d": lh.ravel().tolist(), "hand_right_keypoints_2d": rh.ravel().tolist(),
            "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []}]}))
    return dst


def aberman(style_src: Path, odir: Path) -> dict:
    out = odir / "fixed.bvh"
    if not out.exists():
        r = subprocess.run([sys.executable, "style_transfer/test.py", "--content_src", str(CONTENT["walk"]),
                            "--style_src", str(style_src), "--output_dir", str(odir)], cwd=DME, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-1500:])
    a, names, _ = BVH.load(str(out))
    return metrics(bvh_points(a, names))


# ---------------------------------------------------------------- 실행
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) 참가자 조깅 → CMU BVH (스타일 원천), 실제 걷기 지표
    jog, walk, ik_err, jstats = {}, {}, [], {}
    for p in PEOPLE:
        pts, root, fps = mp_points(MP3D / f"{p}_S1_M06_T01.npz")
        anim = ik(pts, root * UNITS + np.array([0, R.REST_HIP_Y, 0]))
        rep = Animation(anim.rotations.repeat(2, axis=0), anim.positions.repeat(2, axis=0), anim.orients, anim.offsets, anim.parents)
        save(rep, OUT / "style" / f"{p}_jog.bvh", fps * 2)  # Motion Puzzle은 120fps 입력을 [::2]로 60fps로 씀
        back = bvh_points(anim, N31)
        for rot, end, a, b in LIMBS:  # IK 검증: 결과 뼈 방향 vs 목표 방향
            u, v = back[b] - back[a], pts[b] - pts[a]
            ik_err.append(np.degrees(np.arccos(np.clip((nrm(u) * nrm(v)).sum(-1), -1, 1))).mean())
        jog[p] = metrics(pts)
        jstats[p] = stats(channels(pts))
        ws = sorted(MP3D.glob(f"{p}_S1_M01_*T*.npz"))
        wm = [metrics(mp_points(w)[0]) for w in ws]
        walk[p] = {k: float(np.mean([m[k] for m in wm])) for k in METRICS}
    print(f"IK 뼈 방향 오차 평균 {np.mean(ik_err):.2f}°")

    # 2) 콘텐츠(CMU 걷기)
    canim, cnames, cft = BVH.load(str(CONTENT["walk"]))
    cpts = bvh_points(canim, cnames)
    croot = canim.positions[:, 0]
    base = metrics(cpts)

    # 3) 방법 B: 관절 각도 오프셋
    jmean = {k: np.mean([jog[p][k] for p in PEOPLE]) for k in METRICS}
    m3 = {}
    for p in PEOPLE:
        d = {k: jog[p][k] - jmean[k] for k in ("팔꿈치", "무릎", "팔 벌림", "상체 숙임")}
        ratio = jog[p]["팔 흔들림"] / jmean["팔 흔들림"]
        epts = apply_offsets(cpts, d, ratio)
        anim = ik(epts, croot)
        save(anim, OUT / "offset" / f"{p}_walk.bvh", 1.0 / cft)
        m3[p] = metrics(bvh_points(anim, N31))

    # 4) 방법 A: Motion Puzzle
    mp = {}
    for p in PEOPLE:
        style = OUT / "style" / f"{p}_jog.bvh"
        odir = OUT / "puzzle"
        out = odir / f"Style_{p}_jog_Content_{CONTENT['walk'].stem}_fixed.bvh"  # 발 미끄럼 보정판(Motion Puzzle 기본 후처리)
        if not out.exists():
            r = subprocess.run([sys.executable, "test.py", "--content", str(CONTENT["walk"]), "--style", str(style),
                                "--output_dir", str(odir)], cwd=MPZ, capture_output=True, text=True)
            if r.returncode:
                print(r.stderr[-1500:])
                raise SystemExit(f"Motion Puzzle 실패: {p}")
        a, names, _ = BVH.load(str(out))
        mp[p] = metrics(bvh_points(a, names))

    # 4b) 방법 C·D: Aberman 3D(BVH 스타일) / 2D(영상 관절 스타일)
    ab3 = {p: aberman(OUT / "style" / f"{p}_jog.bvh", OUT / "aberman3d" / f"{p}_jog") for p in PEOPLE}
    ab2 = {p: aberman(openpose_dir(MP3D / f"{p}_S1_M06_T01.npz", OUT / "openpose" / f"{p}_jog"), OUT / "aberman2d" / f"{p}_jog")
           for p in PEOPLE}
    # 4c) 방법 E: 관절 각도 AdaIN (절대 / 집단 평균 대비)
    gj = group_stats(list(jstats.values()))
    ad, adr = {}, {}
    for p in PEOPLE:
        for res, grp, tag in ((ad, None, "adain"), (adr, gj, "adain_rel")):
            anim = ik(adain(cpts, jstats[p], grp), croot)
            save(anim, OUT / tag / f"{p}_jog.bvh", 1.0 / cft)
            res[p] = metrics(bvh_points(anim, N31))
    methods = {"오프셋": m3, "AdaIN": ad, "AdaIN상대": adr, "모션퍼즐": mp, "Aberman3D": ab3, "Aberman2D": ab2}

    # 5) 평가
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def ident(src, dst):  # src 결과로 dst(정답 후보) 중 최근접 = 같은 사람인 비율
        A = np.array([[src[p][k] for k in METRICS] for p in PEOPLE])
        B = np.array([[dst[p][k] for k in METRICS] for p in PEOPLE])
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        B = (B - B.mean(0)) / (B.std(0) + 1e-9)
        return float(np.mean([np.argmin(((B - A[i]) ** 2).sum(1)) == i for i in range(len(PEOPLE))]))
    rep = {"base": base, "jog": jog, "walk": walk, "offset": m3, "puzzle": mp, "aberman3d": ab3, "aberman2d": ab2,
           "adain": ad, "adain_rel": adr, "eval": {}}
    print(f"\n콘텐츠(CMU 35_06 걷기) 원래 값: " + "  ".join(f"{k} {v:.0f}°" for k, v in base.items()))
    print("\n지표별 — 사람 간 상관 r (n=12) / 퍼짐 = 사람 간 표준편차(°)")
    print(f"{'':20s}" + "".join(f"{k:>12s}" for k in METRICS))
    rows = [("자연: 조깅↔실제걷기", jog, walk)]
    for name, res in methods.items():
        rows += [(f"{name}: 조깅→결과", jog, res), (f"{name}: 결과↔실제걷기", res, walk)]
    for label, x, y in rows:
        r = [corr([x[p][k] for p in PEOPLE], [y[p][k] for p in PEOPLE]) for k in METRICS]
        rep["eval"][label] = dict(zip(METRICS, r))
        print(f"{label:20s}" + "".join(f"{v:+12.2f}" for v in r))
    for label, res in [("실제 걷기 퍼짐", walk), ("조깅 퍼짐", jog)] + [(f"{n} 결과 퍼짐", r) for n, r in methods.items()]:
        s = [float(np.std([res[p][k] for p in PEOPLE])) for k in METRICS]
        rep["eval"][label] = dict(zip(METRICS, s))
        print(f"{label:20s}" + "".join(f"{v:12.1f}" for v in s))
    for label, a, b in [("실제: 조깅→실제걷기", jog, walk)] + [(f"{n} 결과→실제걷기", r, walk) for n, r in methods.items()] \
            + [(f"{n} 결과→원천조깅", r, jog) for n, r in methods.items()]:
        rep["eval"]["식별 " + label] = ident(a, b)
        print(f"식별 {label}: {ident(a, b):.0%} (우연 {1 / len(PEOPLE):.0%})")
    (OUT / "style_compare.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


def same_motion() -> None:
    """같은 동작에서 뽑은 스타일: 걷기 T01을 원천으로 CMU 걷기에 입히고, 안 쓴 걷기 T02와 비교 (걷기 2테이크 이상 10명)."""
    canim, cnames, cft = BVH.load(str(CONTENT["walk"]))
    cpts, croot = bvh_points(canim, cnames), canim.positions[:, 0]
    ppl = [p for p in PEOPLE if len(sorted(MP3D.glob(f"{p}_S1_M01_*T*.npz"))) >= 2]
    src, held, sst = {}, {}, {}
    for p in ppl:
        w1, w2 = sorted(MP3D.glob(f"{p}_S1_M01_*T*.npz"))[:2]
        pts, root, fps = mp_points(w1)
        anim = ik(pts, root * UNITS + np.array([0, R.REST_HIP_Y, 0]))
        rep = Animation(anim.rotations.repeat(2, axis=0), anim.positions.repeat(2, axis=0), anim.orients, anim.offsets, anim.parents)
        save(rep, OUT / "style" / f"{p}_walk.bvh", fps * 2)
        src[p] = metrics(pts)
        sst[p] = stats(channels(pts))
        held[p] = metrics(mp_points(w2)[0])
    smean = {k: np.mean([src[p][k] for p in ppl]) for k in METRICS}
    m3, mp = {}, {}
    for p in ppl:
        d = {k: src[p][k] - smean[k] for k in ("팔꿈치", "무릎", "팔 벌림", "상체 숙임")}
        anim = ik(apply_offsets(cpts, d, src[p]["팔 흔들림"] / smean["팔 흔들림"]), croot)
        save(anim, OUT / "offset" / f"{p}_walk_from_walk.bvh", 1.0 / cft)
        m3[p] = metrics(bvh_points(anim, N31))
        out = OUT / "puzzle" / f"Style_{p}_walk_Content_{CONTENT['walk'].stem}_fixed.bvh"
        if not out.exists():
            r = subprocess.run([sys.executable, "test.py", "--content", str(CONTENT["walk"]), "--style", str(OUT / "style" / f"{p}_walk.bvh"),
                                "--output_dir", str(OUT / "puzzle")], cwd=MPZ, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stderr[-1500:])
        a, names, _ = BVH.load(str(out))
        mp[p] = metrics(bvh_points(a, names))
    ab3 = {p: aberman(OUT / "style" / f"{p}_walk.bvh", OUT / "aberman3d" / f"{p}_walk") for p in ppl}
    ab2 = {p: aberman(openpose_dir(sorted(MP3D.glob(f"{p}_S1_M01_*T*.npz"))[0], OUT / "openpose" / f"{p}_walk"),
                      OUT / "aberman2d" / f"{p}_walk") for p in ppl}
    gw = group_stats(list(sst.values()))
    ad, adr = {}, {}
    for p in ppl:
        for res, grp, tag in ((ad, None, "adain"), (adr, gw, "adain_rel")):
            anim = ik(adain(cpts, sst[p], grp), croot)
            save(anim, OUT / tag / f"{p}_walk.bvh", 1.0 / cft)
            res[p] = metrics(bvh_points(anim, N31))
    corr = lambda x, y, k: float(np.corrcoef([x[p][k] for p in ppl], [y[p][k] for p in ppl])[0, 1])

    def ident(a, b):
        A = np.array([[a[p][k] for k in METRICS] for p in ppl]); B = np.array([[b[p][k] for k in METRICS] for p in ppl])
        A = (A - A.mean(0)) / (A.std(0) + 1e-9); B = (B - B.mean(0)) / (B.std(0) + 1e-9)
        return float(np.mean([np.argmin(((B - A[i]) ** 2).sum(1)) == i for i in range(len(ppl))]))
    res = {"people": ppl, "src": src, "held": held, "offset": m3, "puzzle": mp, "aberman3d": ab3, "aberman2d": ab2,
           "adain": ad, "adain_rel": adr, "eval": {}}
    print(f"\n같은 동작 원천(걷기 T01 → CMU 걷기, 비교 대상 = 안 쓴 걷기 T02), n={len(ppl)}")
    print(f"{'':20s}" + "".join(f"{k:>12s}" for k in METRICS))
    for label, x, y in (("상한: 걷기T01↔T02", src, held), ("오프셋 결과↔T02", m3, held), ("AdaIN 결과↔T02", ad, held),
                        ("AdaIN상대 결과↔T02", adr, held), ("모션퍼즐 결과↔T02", mp, held),
                        ("Aberman3D 결과↔T02", ab3, held), ("Aberman2D 결과↔T02", ab2, held)):
        r = [corr(x, y, k) for k in METRICS]
        res["eval"][label] = dict(zip(METRICS, r))
        print(f"{label:20s}" + "".join(f"{v:+12.2f}" for v in r))
    for label, a in (("상한: 걷기T01→T02", src), ("오프셋 결과→T02", m3), ("AdaIN 결과→T02", ad), ("AdaIN상대 결과→T02", adr),
                     ("모션퍼즐 결과→T02", mp), ("Aberman3D 결과→T02", ab3), ("Aberman2D 결과→T02", ab2)):
        res["eval"]["식별 " + label] = ident(a, held)
        print(f"식별 {label}: {ident(a, held):.0%} (우연 {1 / len(ppl):.0%})")
    (OUT / "style_compare_same.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    same_motion() if "--same" in sys.argv else main()
