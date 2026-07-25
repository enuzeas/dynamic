"""세션 4-0a — GVHMR/FootMR `hmr4d_results.pt`(smpl_params_global) -> retarget_smpl_to_cmu.py 입력용 npz.

**2026-07-25 확인: 원래 계획(smpl2bvh 사용)을 뒤집은 근거.**
원래 계획은 smpl2bvh(KosukeFukazawa)로 "SMPL 24관절 BVH"를 만들고 4-0b가 그걸 읽는 것이었다.
실제로 clone해서 돌려본 결과 이 단계 자체가 불필요하다는 게 드러났다:

1. FootMR 소스(`external/FootMR/hmr4d/model/footmr/pipeline/footmr_pipeline.py`,
   `footmr_pl_demo.py`)를 직접 읽어 `smpl_params_global` 딕셔너리의 실제 키·shape을 확인함
   — `global_orient`(F,3), `body_pose`(F,63 = 21관절 축각), `transl`(F,3). 이 body_pose 21관절
   순서가 `retarget_smpl_to_cmu.py`의 `SMPL_JOINT_NAMES[1:22]`와 정확히 같은 표준 SMPL 순서다.
   즉 GVHMR/FootMR 출력은 이미 "로컬 축각 회전"이라 forward kinematics도 SMPL 본체 파일도
   필요 없이 그대로 이어붙이면 된다.
2. smpl2bvh를 실제로 실행해 검증하는 과정에서 그쪽 자체의 버그를 발견함:
   `external/smpl2bvh/utils/quat.py`의 `from_axis_angle()`이 회전이 정확히 0인 관절에서
   0/0 나눗셈으로 NaN을 낸다(합성 데이터로 실측 확인 — 정지 관절이 하나라도 있으면 발생,
   즉 보통 모션 대부분에서 발생). `retarget_smpl_to_cmu.py`가 쓰는 Quaternions.from_angle_axis
   (motion_puzzle 자산)는 이미 0벡터를 안전하게 처리해서 이 문제가 없다.

그래서 smpl2bvh는 쓰지 않는다(clone은 external/에 참고·시각 디버그용으로만 남겨둠, 파이프라인
제외). SMPL 24관절 중 손(left_hand=22, right_hand=23)은 GVHMR/FootMR 출력에 아예 없음(SMPL-X는
손가락을 별도 파라미터로 다룸) — 항등회전(0)으로 채운다. 손가락 디테일은 원래 스코프 밖
(`samsam_shooting_guide.md`).

사용법 (motion_puzzle 또는 mcmldm conda env 아무 곳이나 — torch만 있으면 됨):
  conda run -n motion_puzzle python hmr4d_to_npz.py \
      --pt outputs/demo/xxx/hmr4d_results.pt --out smpl_pose.npz --fps 60
그 뒤 그대로 4-0b 입력으로 쓴다:
  conda run -n motion_puzzle python retarget_smpl_to_cmu.py --npz smpl_pose.npz --out out.bvh
"""
from __future__ import annotations

import numpy as np
import torch


def smpl_params_global_to_rotations(smpl_params_global: dict) -> tuple[np.ndarray, np.ndarray]:
    """global_orient(F,3)+body_pose(F,63) -> rotations(F,24,3) 축각, trans(F,3, CMU-BVH cm 스케일).

    GVHMR/FootMR의 transl은 SMPL 관례대로 미터 단위인데, `retarget_smpl_to_cmu.py`가 쓰는
    CMU BVH 골격(`RAW31_OFFSETS`)은 센티미터 스케일(힙 높이 ~100 단위) — 여기서 ×100 해서
    맞춘다. 안 하면 캐릭터가 실제 크기의 1/100로 뷰어에 나온다.
    """
    global_orient = smpl_params_global["global_orient"].detach().cpu().numpy()
    body_pose = smpl_params_global["body_pose"].detach().cpu().numpy()
    transl = smpl_params_global["transl"].detach().cpu().numpy()

    n_frames = global_orient.shape[0]
    rotations = np.zeros((n_frames, 24, 3))
    rotations[:, 0] = global_orient
    rotations[:, 1:22] = body_pose.reshape(n_frames, 21, 3)
    return rotations, transl * 100.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="GVHMR/FootMR hmr4d_results.pt 경로")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, required=True, help="원본 촬영 영상 fps(파일에 없음, 직접 지정)")
    args = parser.parse_args()

    pred = torch.load(args.pt, map_location="cpu")
    rotations, trans = smpl_params_global_to_rotations(pred["smpl_params_global"])
    np.savez(args.out, rotations=rotations, trans=trans, fps=args.fps)
    print(f"저장 완료: {args.out} ({rotations.shape[0]}프레임)")
