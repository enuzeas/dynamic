"""'걷다가 점프 / 기기 / 춤' 런웨이 클립 만들기 (STyMo 결과 확인용).

앞부분은 중립 걷기, 뒷부분은 점프·기어가기·춤. 뒷부분을 (a) 중립 그대로 (b) STyMo 화난 스타일(강도 1.0)
로 각각 이어 붙여 '원본' / '스타일' 두 파일을 만든다.

입력은 bvh_zup_to_yup.py 로 Y-up 변환한 BVH(work_yup/). 뷰어는 Root 관절을 안 읽고 Hips만 읽으므로
  - Root 의 방향(yaw)은 Hips 회전에 구워 넣는다(각 구간 첫 프레임 방향을 정면으로 정규화).
  - Root 의 이동(앞으로 걷는 거리)은 버린다 → 제자리 재생(카메라에 항상 잡힘).
이어 붙이는 지점은 N프레임 동안 관절 회전을 slerp, Hips 위치를 lerp 로 섞는다.
"""
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUP = os.path.join(BASE, "work_yup")
OUT = os.path.join(BASE, "viewer_try", "viewer_data")

FPS = 60
BLEND_FRAMES = 36        # 0.6초
WALK_START, WALK_LEN = 120, 420   # 걷기 7초
# 동작별 (이름, 시작 프레임, 길이): 앞쪽 가만히 서 있는 구간은 건너뛰고 본 동작부터 쓴다
CLIPS = {
    "jump": ("Jump", 210, 450),          # 3.5초부터 7.5초
    "crawl": ("Loco_Crawling", 90, 720),  # 1.5초부터 12초
    "dance": ("Dance", 240, 720),        # 4초부터 12초
}


def load(path):
    L = open(path).read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("MOTION"))
    n = int(L[i + 1].split()[1])
    data = np.array([[float(x) for x in l.split()] for l in L[i + 3 : i + 3 + n]])
    return L[:i], data


def bake(data):
    """Root yaw -> Hips 회전, Root 이동 제거. 각 구간 첫 프레임이 정면(+Z)."""
    rroot = R.from_euler("ZYX", data[:, 3:6], degrees=True)
    f = rroot.apply([0, 0, 1])
    yaw = np.arctan2(f[:, 0], f[:, 2])
    corr = R.from_rotvec(np.stack([np.zeros_like(yaw), yaw - yaw[0], np.zeros_like(yaw)], axis=1))
    out = data.copy()
    out[:, 0:6] = 0.0
    out[:, 6:9] = corr.apply(data[:, 6:9])
    out[:, 9:12] = (corr * R.from_euler("ZYX", data[:, 9:12], degrees=True)).as_euler("ZYX", degrees=True)
    return out


def smoothstep(t):
    return t * t * (3 - 2 * t)


def join(a, b, n):
    """a 의 마지막 n프레임과 b 의 처음 n프레임을 섞어 하나로 잇는다."""
    w = smoothstep((np.arange(n) + 1.0) / (n + 1.0))
    a_tail, b_head = a[-n:], b[:n]
    mix = np.zeros_like(a_tail)
    mix[:, 6:9] = (1 - w)[:, None] * a_tail[:, 6:9] + w[:, None] * b_head[:, 6:9]
    for c in range(9, a.shape[1], 3):
        r1 = R.from_euler("ZYX", a_tail[:, c : c + 3], degrees=True)
        r2 = R.from_euler("ZYX", b_head[:, c : c + 3], degrees=True)
        delta = (r1.inv() * r2).as_rotvec() * w[:, None]
        mix[:, c : c + 3] = (r1 * R.from_rotvec(delta)).as_euler("ZYX", degrees=True)
    return np.vstack([a[:-n], mix, b[n:]])


def write(header, data, path):
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write(f"MOTION\nFrames: {len(data)}\nFrame Time: {1.0 / FPS:.6f}\n")
        for row in data:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def main():
    header, walk = load(os.path.join(YUP, "Loco_Walk_neutral.bvh"))
    walk_seg = bake(walk)[WALK_START : WALK_START + WALK_LEN]
    for key, (name, a0, alen) in CLIPS.items():
        for kind in ("neutral", "styled"):
            _, act = load(os.path.join(YUP, f"{name}_{kind}.bvh"))
            seq = join(walk_seg, bake(act)[a0 : a0 + alen], BLEND_FRAMES)
            out_name = "original" if kind == "neutral" else "styled"
            path = os.path.join(OUT, f"runway_{key}_{out_name}.bvh")
            write(header, seq, path)
            print(f"{path}  {len(seq)} frames ({len(seq) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
