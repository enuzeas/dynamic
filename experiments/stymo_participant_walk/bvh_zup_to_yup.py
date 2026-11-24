"""STyMo(MOCHA) BVH의 Hips 이하 뼈대를 Y-up 기준으로 다시 표현한다 (viewer_try 확인용 변환기).

이 BVH는 Hips 이하 관절의 로컬 좌표가 Z-up(척추 +Z, 다리 -Z, 왼쪽 -X)이고, Hips 회전값에
Z-up -> Y-up 변환(H0 = Rz(180)·Rx(90))이 섞여 있다. 뷰어는 모든 관절의 rest 방향이 Y-up이라고
가정하므로, 같은 동작을 유지한 채 좌표만 바꾼다.

  Hips 아래 모든 OFFSET:  o' = H0 · o
  Hips 회전:               R_h' = R_h · H0⁻¹
  그 아래 모든 관절 회전:   R'   = H0 · R · H0⁻¹
(루트·Hips 위치 채널과 Hips OFFSET은 Root 좌표 기준이라 그대로 둔다.)

사용: python bvh_zup_to_yup.py in.bvh out.bvh
"""
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

H0 = R.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])  # Rz(180)·Rx(90)
H0_INV = H0.inv()


def convert(src, dst):
    text = open(src).read().split("\n")
    m_idx = next(i for i, l in enumerate(text) if l.startswith("MOTION"))
    header = text[:m_idx]

    joints = []  # (name, n_channels, order_tuple)
    depth_names = []
    under_hips = False
    hips_depth = None
    out_header = []
    cur = None
    depth = 0
    in_end_site = False
    for l in header:
        s = l.strip()
        if s.startswith(("ROOT", "JOINT")):
            cur = s.split()[1]
            joints.append([cur, 0, None, False])
        if s.startswith("End Site"):
            in_end_site = True
        if s == "{":
            depth += 1
            depth_names.append(("END" if in_end_site else cur, depth))
            if cur == "Hips" and not in_end_site and hips_depth is None:
                hips_depth = depth
        if s == "}":
            if in_end_site:
                in_end_site = False
            if hips_depth is not None and depth == hips_depth:
                hips_depth = None
            depth -= 1
            depth_names.pop()
        m = re.match(r"\s*CHANNELS\s+(\d+)\s+(.*)", l)
        if m:
            joints[-1][1] = int(m.group(1))
            joints[-1][2] = tuple(m.group(2).split())
        m = re.match(r"(\s*)OFFSET\s+(\S+)\s+(\S+)\s+(\S+)(.*)", l)
        if m:
            # Hips 자신의 OFFSET은 그대로, Hips의 자식(깊이 > hips_depth)과 End Site는 H0 적용
            inside_hips_subtree = hips_depth is not None and depth > hips_depth
            if inside_hips_subtree:
                o = H0.apply([float(m.group(2)), float(m.group(3)), float(m.group(4))])
                l = f"{m.group(1)}OFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}{m.group(5)}"
        out_header.append(l)

    # 'Hips 서브트리' 표시: 이름 목록에서 Hips 이후 모든 관절
    names = [j[0] for j in joints]
    hips_i = names.index("Hips")
    for k, j in enumerate(joints):
        j[3] = k > hips_i  # Hips 아래(문서 순서상 이후) 관절

    frame_lines = [l for l in text[m_idx:] if l.strip()]
    n_frames = int(frame_lines[1].split()[1])
    frame_time = frame_lines[2]
    data = np.array([[float(x) for x in l.split()] for l in frame_lines[3 : 3 + n_frames]])

    order_ok = all(j[2][-3:] == ("Zrotation", "Yrotation", "Xrotation") for j in joints)
    assert order_ok, "회전 채널 순서가 Z,Y,X가 아닌 관절이 있다"

    out = data.copy()
    p = 0
    for name, n_ch, order, below in joints:
        rot_slice = slice(p + n_ch - 3, p + n_ch)
        eul = data[:, rot_slice]  # (Z, Y, X) 도
        rot = R.from_euler("ZYX", eul, degrees=True)
        if name == "Hips":
            new = rot * H0_INV
        elif below:
            new = H0 * rot * H0_INV
        else:
            new = rot
        out[:, rot_slice] = new.as_euler("ZYX", degrees=True)
        p += n_ch

    with open(dst, "w") as f:
        f.write("\n".join(out_header) + "\n")
        f.write(f"MOTION\nFrames: {n_frames}\n{frame_time}\n")
        for row in out:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    print(f"ok {src} -> {dst}  ({n_frames} frames, {len(joints)} joints)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
