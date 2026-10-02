"""GVHMR/FootMR hmr4d_results.pt → reports/gvhmr/{클립}.npz — style_compare·invariant_traits를 GVHMR 3D로 다시 돌리기 위한 입력.

SMPL-X 순운동학으로 몸 22관절의 월드 위치(미터, Y 위)를 구한다. 체형은 클립 평균 betas 하나로 고정한다.
같은 클립의 MediaPipe npz(reports/mp3d)와 이름을 맞춰 두면, POSE_SRC=gvhmr 일 때 style_compare.mp_points가
'joints' 키를 보고 이 경로를 탄다. Colab 배치(/content/batch.py)는 GVHMR 학습 조건에 맞춰 59.94 → 29.97fps로 줄여
넣으므로, 여기서 다시 2배 보간해 59.94fps로 저장한다.

  conda run -n motion_puzzle python gvhmr_to_npz.py <드라이브 samsam_gvhmr/outputs 폴더> [--fps 29.97]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SMPLX = ROOT / "SMPL" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]  # SMPL-X 몸 22관절


def joints(sp: dict, model) -> np.ndarray:
    """smpl_params_global(numpy) → (F,22,3) 월드 관절 위치."""
    v = model["v_template"] + model["shapedirs"][:, :, :10] @ sp["betas"].mean(0)
    rest = model["J_regressor"][:22] @ v
    F = len(sp["global_orient"])
    aa = np.concatenate([sp["global_orient"][:, None], sp["body_pose"].reshape(F, 21, 3)], 1)
    rot = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(F, 22, 3, 3)
    G, P = np.zeros((F, 22, 3, 3)), np.zeros((F, 22, 3))
    G[:, 0], P[:, 0] = rot[:, 0], rest[0] + sp["transl"]
    for j in range(1, 22):
        p = PARENTS[j]
        G[:, j] = G[:, p] @ rot[:, j]
        P[:, j] = P[:, p] + G[:, p] @ (rest[j] - rest[p])
    return P


def demo() -> None:
    """자체 확인: 항등 자세면 쉴 때 관절 그대로, 뼈 길이는 어떤 회전에서도 그대로."""
    model = np.load(SMPLX)
    F = 4
    sp = {"betas": np.zeros((F, 10)), "global_orient": np.zeros((F, 3)), "body_pose": np.zeros((F, 63)), "transl": np.zeros((F, 3))}
    rest = model["J_regressor"][:22] @ model["v_template"]
    assert np.allclose(joints(sp, model)[0], rest)
    sp["body_pose"] = np.random.default_rng(0).normal(0, 0.5, (F, 63))
    P = joints(sp, model)
    bone = lambda X: np.linalg.norm(X[..., 1:, :] - X[..., PARENTS[1:], :], axis=-1)
    assert np.allclose(bone(P), bone(rest)[None]), "뼈 길이가 바뀜"
    assert rest[1, 0] > rest[2, 0], "SMPL-X 왼쪽 엉덩이가 +X가 아님"
    print("gvhmr_to_npz demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="outputs 폴더 (클립별 하위 폴더에 hmr4d_results.pt)")
    ap.add_argument("--fps", type=float, default=30000 / 1001)
    a = ap.parse_args()
    if not a.src:
        demo()
        raise SystemExit
    import torch

    model, dst = np.load(SMPLX), ROOT / "reports" / "gvhmr"
    dst.mkdir(parents=True, exist_ok=True)
    for pt in sorted(Path(a.src).glob("*/hmr4d_results.pt")):
        sp = {k: v.numpy().astype(np.float64) for k, v in torch.load(pt, map_location="cpu")["smpl_params_global"].items()}
        P = joints(sp, model)
        # 2배로 선형 보간해 MediaPipe npz와 같은 59.94fps로 맞춘다 (style_compare가 Motion Puzzle에 60fps 기준으로 넘김)
        P2 = np.empty((2 * len(P) - 1, *P.shape[1:]))
        P2[::2], P2[1::2] = P, (P[:-1] + P[1:]) / 2
        np.savez(dst / f"{pt.parent.name.replace('_CAM_F', '')}.npz", joints=P2, fps=2 * a.fps)  # mp3d와 같은 이름
        print(pt.parent.name, len(sp["transl"]))
