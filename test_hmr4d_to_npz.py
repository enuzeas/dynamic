"""hmr4d_to_npz 자체 검증 — 실제 GVHMR/FootMR 없이, `hmr4d_results.pt`와 같은 구조의 합성
`smpl_params_global`(global_orient/body_pose/transl, torch 텐서)로 (1) 변환된 npz의 shape·NaN
여부, (2) 그 npz가 retarget_smpl_to_cmu.py -> Motion Puzzle까지 그대로 먹히는지(제일 중요한
통합 검증) 확인한다.

실행 (motion_puzzle conda env 필요 — torch로 합성 pt 생성 + Animation/Quaternions/BVH/test.py
재사용 때문):
  conda run -n motion_puzzle python test_hmr4d_to_npz.py
"""
from __future__ import annotations

import numpy as np
import torch

from hmr4d_to_npz import smpl_params_global_to_rotations
from test_retarget_smpl_to_cmu import check_motion_puzzle_integration, check_roundtrip
from retarget_smpl_to_cmu import save_bvh


def make_synthetic_hmr4d_results(n_frames: int = 60) -> dict:
    """실제 hmr4d_results.pt와 같은 키 구조: {"smpl_params_global": {global_orient, body_pose, transl}}."""
    t = torch.linspace(0, 2 * 3.14159265, n_frames)
    body_pose = torch.zeros(n_frames, 63)
    body_pose[:, 18 * 3 + 2] = 0.8 * torch.sin(t)  # right_elbow(SMPL_JOINT_NAMES idx19 = body_pose 관절18)를 z축으로 굽힘

    global_orient = torch.zeros(n_frames, 3)
    transl = torch.zeros(n_frames, 3)
    transl[:, 1] = 1.0
    transl[:, 2] = torch.linspace(0, 0.2, n_frames)

    return {"smpl_params_global": {"global_orient": global_orient, "body_pose": body_pose, "transl": transl}}


def check_conversion(rotations: np.ndarray, trans: np.ndarray, n_frames: int) -> None:
    assert rotations.shape == (n_frames, 24, 3)
    assert trans.shape == (n_frames, 3)
    assert not np.isnan(rotations).any(), "변환된 rotations에 NaN이 있다"
    assert np.array_equal(rotations[:, 22], np.zeros((n_frames, 3))), "손(22) 항등회전이 아니다"
    assert np.array_equal(rotations[:, 23], np.zeros((n_frames, 3))), "손(23) 항등회전이 아니다"
    assert not np.array_equal(rotations[:, 19], np.zeros((n_frames, 3))), "right_elbow가 안 움직였다"
    print("  변환 검증 통과 (24관절, NaN 없음, 손 항등회전 유지, right_elbow 반영)")


def main() -> None:
    n_frames = 60
    pred = make_synthetic_hmr4d_results(n_frames)

    rotations, trans = smpl_params_global_to_rotations(pred["smpl_params_global"])
    check_conversion(rotations, trans, n_frames)

    out_path = "/tmp/hmr4d_to_npz_test_out.bvh"
    save_bvh(rotations, trans, out_path, fps=60.0)  # trans는 smpl_params_global_to_rotations가 이미 cm로 변환함
    print(f"합성 hmr4d_results -> BVH 저장: {out_path}")

    check_roundtrip(out_path)
    check_motion_puzzle_integration(out_path)

    print("자체 검증 통과")


if __name__ == "__main__":
    main()
