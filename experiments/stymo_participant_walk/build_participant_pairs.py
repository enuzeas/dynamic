"""STyMo 학습 입력 쌍 만들기 — 공통 걷기 콘텐츠(neutral) → 참가자 걷기(style), 참가자별 1쌍.

기존 코드는 수정하지 않는다. 재사용:
  - 팀 코드 `style_compare`(mp_points / bvh_points / between / frame / mat2quat)와 `run_style_pilot.load()`
  - STyMo 샘플 BVH(`motion/mocha/happy/source/neutral_to_happy.bvh`)의 HIERARCHY를 그대로 복사해 스켈레톤·채널 규약을 맞춘다.

변환 (AdaIN 경로의 `ik()`와 같은 방식: 3D 관절점의 뼈 방향 → 전역 회전 → 로컬 회전):
  STyMo 스켈레톤은 Hips 아래 로컬이 Z-up(척추 +Z, 앞 +Y, 왼쪽 -X)이고 Hips 회전에 H0=Rz(180)·Rx(90)이 섞여 있다
  (tools/bvh_zup_to_yup.py 주석과 같은 규약). 그래서 Y-up 등가 뼈대(offset' = H0·offset)에서 방향 IK로 G_yup을 구한 뒤
    G_native = G_yup · H0,   L_native = H0⁻¹ · L_yup · H0   (Hips는 G_native 그대로)
  로 옮긴다. 관절 대응은 이름 그대로다(LeftUpLeg↔Lhip→Lknee 등). 손끝·발끝·머리 끝은 로컬 항등(부모 회전 상속).
  척추: Hips는 골반 프레임, Spine1에 가슴 프레임을 주고 Spine/Spine2/Spine3과 어깨는 상속(CMU ik()와 같은 배치).

루트/방향/높이 (소스·타깃에 같은 절차):
  - 몸 정면(across×up)의 클립 평균을 +Z로 돌리는 Y축 회전을 점들과 루트 경로에 같이 적용(정면 정규화).
  - Root 위치 = 수평 경로(시작점을 원점으로), Hips 위치 = 샘플 Hips offset + (0, 평균을 뺀 상하 움직임, 0). 마지막에 FK로 발/발끝 최저점을
    바닥(y=0)에 맞추는 상수 이동 하나만 적용(`R.ground()`와 같은 성격). 발 고정(foot lock)은 없다.
  - 단위 미터, 프레임 간격 1/60초. 참가자 NPZ는 59.94fps지만 프레임 수를 보존하려고 60fps로 기록한다(0.1% 차이).
참가자 루트는 기존 `mp_points()`의 추정치(몸통 크기로 깊이를 추정)를 그대로 쓴다.

출력: ~/work/stymo_trial/participant_pairs/<P00X>/{source,target}/commonwalk_to_<P00X>.bvh  (이미 있으면 중단)

실행(기존 환경 재사용):
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/work/stymo_trial/tools/build_participant_pairs.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

if os.environ.get("NUMPY_EXPERIMENTAL_DTYPE_API") != "1":
    raise SystemExit("NUMPY_EXPERIMENTAL_DTYPE_API=1 을 붙여 실행해야 한다 (docstring 참고)")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

sys.path.insert(0, str(Path.home() / "Documents" / "dynamic_data" / "scripts"))
import run_style_pilot as X  # noqa: E402  (style_compare를 X.S로 불러온다)

S = X.S
STY = Path.home() / "work" / "stymo_trial"
TEMPLATE = STY / "STyMo" / "motion" / "mocha" / "happy" / "source" / "neutral_to_happy.bvh"
CONTENT = Path("공통_걷기.bvh")  # 공통 걷기 모캡 BVH(별도 확보)의 실제 경로로 바꿔 쓴다
OUT = STY / "participant_pairs"
FRAME_TIME = 1.0 / 60.0
PAIRS = [  # (참가자, source 콘텐츠 구간, 스타일 NPZ 모션 라벨, 스타일 구간)
    ("P001", (9020, 9200), "Walking-M01", (30, 210)),
    ("P002", (9032, 9200), "Walking", (48, 216)),
    ("P003", (9020, 9200), "Walking", (228, 408)),
]
H0 = np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])  # Rz(180)·Rx(90), bvh_zup_to_yup.py와 같은 값

# (회전 관절, 그 뼈의 쉴 때 끝 관절, 목표 시작점, 목표 끝점)
LIMBS = [
    ("LeftUpLeg", "LeftLeg", "Lhip", "Lknee"), ("LeftLeg", "LeftFoot", "Lknee", "Lank"), ("LeftFoot", "LeftToeBase", "Lank", "Ltoe"),
    ("RightUpLeg", "RightLeg", "Rhip", "Rknee"), ("RightLeg", "RightFoot", "Rknee", "Rank"), ("RightFoot", "RightToeBase", "Rank", "Rtoe"),
    ("Neck1", "Head", "neck", "head"),
    ("LeftArm", "LeftForeArm", "Lsh", "Lel"), ("LeftForeArm", "LeftHand", "Lel", "Lwr"),
    ("RightArm", "RightForeArm", "Rsh", "Rel"), ("RightForeArm", "RightHand", "Rel", "Rwr"),
]


def parse_template(path: Path = TEMPLATE):
    """HIERARCHY 텍스트와 (이름, 부모, offset, 채널 수)를 읽는다. End Site는 관절로 세지 않는다."""
    text = path.read_text()
    header = text[: text.index("MOTION")]
    names, parents, offs, nch, stack = [], [], [], [], []
    last, in_end = None, False
    for line in header.splitlines():
        s = line.strip()
        m = re.match(r"(ROOT|JOINT)\s+(\S+)", s)
        if m:
            names.append(m.group(2)); parents.append(stack[-1] if stack else -1); offs.append(None); nch.append(0)
            last = len(names) - 1
        elif s.startswith("End Site"):
            in_end = True
        elif s == "{":
            stack.append(last) if not in_end else None
        elif s == "}":
            if in_end:
                in_end = False
            else:
                stack.pop()
        elif s.startswith("OFFSET") and not in_end:
            offs[last] = [float(v) for v in s.split()[1:4]]
        elif s.startswith("CHANNELS"):
            nch[last] = int(s.split()[1])
    return header, names, np.array(parents), np.array(offs), nch


HEADER, NAMES, PAR, OFF, NCH = parse_template()
NJ = len(NAMES)
IX = {n: i for i, n in enumerate(NAMES)}
assert NJ == 25 and NCH[0] == 6 and NCH[1] == 6 and all(c == 3 for c in NCH[2:]), "예상과 다른 STyMo 스켈레톤"
CHAN_LINE = "Zrotation Yrotation Xrotation"
assert all(CHAN_LINE in l for l in HEADER.splitlines() if "CHANNELS" in l), "회전 채널 순서가 ZYX가 아님"


def fk_positions(local_rot: np.ndarray, root_pos: np.ndarray, hips_pos: np.ndarray) -> np.ndarray:
    """로컬 회전 (T,25,3,3), Root 위치 (T,3), Hips 위치 (T,3) → 세계 위치 (T,25,3) (네이티브 규약 그대로)."""
    T = len(local_rot)
    G = np.empty((T, NJ, 3, 3)); P = np.empty((T, NJ, 3))
    G[:, 0] = local_rot[:, 0]; P[:, 0] = root_pos
    for j in range(1, NJ):
        p = PAR[j]
        G[:, j] = G[:, p] @ local_rot[:, j]
        P[:, j] = hips_pos if j == 1 else P[:, p] + (G[:, p] @ OFF[j][:, None])[..., 0]
    return P


def heading_yaw(pts: dict) -> float:
    """정면(across×up) 평균을 +Z로 돌리는 Y축 회전각 (CMU 축: 위 +Y, 앞 +Z, 왼쪽 +X)."""
    f = np.cross(pts["Lhip"] - pts["Rhip"], pts["neck"] - pts["pelvis"])
    f = f[:, [0, 2]] / (np.linalg.norm(f[:, [0, 2]], axis=1, keepdims=True) + 1e-12)
    m = f.mean(0)
    return float(np.arctan2(m[0], m[1]))  # 정면이 +Z에서 x쪽으로 벗어난 각


def rotate_y(v: np.ndarray, ang: float) -> np.ndarray:
    """v를 Y축 기준 -ang 만큼 돌린다 (정면 → +Z)."""
    c, s = np.cos(-ang), np.sin(-ang)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return v @ Ry.T


def build(pts: dict, root_m: np.ndarray):
    """pts: CMU 축 관절점(방향만 사용), root_m: (T,3) 미터. → (BVH 채널 값 (T, 6+6+3*23), 진단 dict)"""
    T = len(root_m)
    yaw = heading_yaw(pts)
    pts = {k: rotate_y(v, yaw) for k, v in pts.items()}
    root = rotate_y(root_m, yaw)

    off_y = OFF.copy()
    off_y[2:] = OFF[2:] @ H0.T  # Hips 아래 offset을 Y-up으로 (Root·Hips offset은 Root 좌표 그대로)
    rest = np.zeros((NJ, 3))
    for j in range(1, NJ):
        rest[j] = rest[PAR[j]] + off_y[j]
    mid = lambda a, b: (rest[IX[a]] + rest[IX[b]]) / 2
    rest_up = mid("LeftArm", "RightArm") - rest[IX["Hips"]]
    hips_r = S.frame(rest_up, rest[IX["LeftUpLeg"]] - rest[IX["RightUpLeg"]])
    chest_r = S.frame(rest_up, rest[IX["LeftArm"]] - rest[IX["RightArm"]])
    hips_t = S.frame(pts["neck"] - pts["pelvis"], pts["Lhip"] - pts["Rhip"])
    chest_t = S.frame(pts["neck"] - pts["pelvis"], pts["Lsh"] - pts["Rsh"])

    G = np.tile(np.eye(3), (T, NJ, 1, 1))  # Y-up 등가 뼈대의 전역 회전 (Root는 항등)
    G[:, IX["Hips"]] = hips_t @ hips_r.T
    limb = {r: (e, a, b) for r, e, a, b in LIMBS}
    for j in range(2, NJ):
        G[:, j] = G[:, PAR[j]]
        if NAMES[j] == "Spine1":
            G[:, j] = chest_t @ chest_r.T
        if NAMES[j] in limb:
            end, a, b = limb[NAMES[j]]
            v = (G[:, PAR[j]] @ (rest[IX[end]] - rest[j])[:, None])[..., 0]
            G[:, j] = S.between(v, pts[b] - pts[a]) @ G[:, PAR[j]]
    Gn = G @ H0  # 네이티브 규약의 전역 회전
    L = np.empty_like(Gn)
    L[:, 0] = np.eye(3)
    L[:, 1] = Gn[:, 1]
    for j in range(2, NJ):
        L[:, j] = np.swapaxes(Gn[:, PAR[j]], -1, -2) @ Gn[:, j]

    # 위치: Root = 수평 경로, Hips = 샘플 Hips offset + 평균을 뺀 상하 움직임, 마지막에 상수 이동으로 접지
    root_pos = np.stack([root[:, 0] - root[0, 0], np.zeros(T), root[:, 2] - root[0, 2]], axis=1)
    hips_pos = np.tile(OFF[1], (T, 1)); hips_pos[:, 1] += root[:, 1] - root[:, 1].mean()
    foot = [IX[n] for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    P = fk_positions(L, root_pos, hips_pos)
    lift = -P[:, foot, 1].min()
    hips_pos[:, 1] += lift
    P = fk_positions(L, root_pos, hips_pos)

    eul = Rot.from_matrix(L[:, 2:].reshape(-1, 3, 3)).as_euler("ZYX", degrees=True).reshape(T, NJ - 2, 3)
    eul0 = Rot.from_matrix(L[:, 1]).as_euler("ZYX", degrees=True)
    vals = np.concatenate([root_pos, np.zeros((T, 3)), hips_pos, eul0, eul.reshape(T, -1)], axis=1)
    assert vals.shape[1] == 6 + 6 + 3 * (NJ - 2)

    # 진단: 결과 뼈 방향 vs 목표 방향 (정면 정규화 후 좌표계에서)
    ixp = {"Lhip": "LeftUpLeg", "Lknee": "LeftLeg", "Lank": "LeftFoot", "Ltoe": "LeftToeBase", "Rhip": "RightUpLeg", "Rknee": "RightLeg",
           "Rank": "RightFoot", "Rtoe": "RightToeBase", "Lsh": "LeftArm", "Lel": "LeftForeArm", "Lwr": "LeftHand", "Rsh": "RightArm",
           "Rel": "RightForeArm", "Rwr": "RightHand", "neck": None, "head": "Head"}
    errs = {}
    for r, e, a, b in LIMBS:
        got = P[:, IX[e]] - P[:, IX[r]]
        want = pts[b] - pts[a]
        c = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1) + 1e-12)
        errs[f"{r}->{e}"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))).max())
    return vals, {"yaw_deg": float(np.degrees(yaw)), "lift_m": float(lift), "max_dir_err_deg": errs}, P


def write_bvh(path: Path, vals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in vals)
    with open(path, "x", newline="\n") as f:  # 덮어쓰지 않음
        f.write(HEADER + f"MOTION\nFrames: {len(vals)}\nFrame Time: {FRAME_TIME:.6f}\n" + body + "\n")


def main() -> None:
    for d in (OUT, STY):
        if str(d.resolve()).startswith(str(X.WORKTREE.resolve())):
            raise SystemExit(f"저장소 안 경로 금지: {d}")
    if OUT.exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {OUT}")
    for p in (CONTENT, TEMPLATE, *(X.MP3D / f"P00{i} {m}.npz" for i, m in ((1, "Walking-M01"), (2, "Walking"), (3, "Walking")))):
        if not p.exists():
            raise SystemExit(f"입력 없음: {p}")
    full, fnames, _ = S.BVH.load(str(CONTENT))  # 전체를 읽어 [a:b]로 자른다(BVH.load의 end 규약 문제를 피함)
    for pid, crng, motion, srng in PAIRS:
        a, b = crng
        sub, _, _ = S.BVH.load(str(CONTENT), a, b + 1)  # BVH.load는 start <= i < end-1 만 읽는다
        assert len(sub.rotations) == b - a, f"콘텐츠 프레임 수 {len(sub.rotations)} != {b - a}"
        assert np.allclose(sub.positions, full.positions[a:b]) and np.allclose(sub.rotations.qs, full.rotations.qs[a:b]), "창 읽기가 전체 슬라이스와 다름"
        cpts = S.bvh_points(sub, fnames)
        croot = (sub.positions[:, 0] - sub.positions[0, 0] * np.array([1, 0, 1])) / 100.0  # cm → m
        spts, sroot, sfps = X.load(pid, motion, srng)
        assert len(croot) == len(sroot), f"{pid}: 프레임 수 불일치 {len(croot)} vs {len(sroot)}"
        name = f"commonwalk_to_{pid}.bvh"
        for kind, pts, root in (("source", cpts, croot), ("target", spts, sroot)):
            vals, diag, _ = build(pts, root)
            write_bvh(OUT / pid / kind / name, vals)
            print(f"{pid} {kind}: {len(vals)} frames  yaw {diag['yaw_deg']:.1f} deg  max dir err {max(diag['max_dir_err_deg'].values()):.2f} deg")


if __name__ == "__main__":
    main()
