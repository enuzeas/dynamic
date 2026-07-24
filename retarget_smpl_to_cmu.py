"""세션 4-0b — SMPL(24관절) 로컬 회전 → Motion Puzzle `--content` 입력용 CMU 원본 31관절 BVH 리타겟.

**2026-07-23 실제 테스트로 잡은 함정**: 처음엔 Motion Puzzle이 "쓰는" 21관절 골격을
그대로 BVH로 저장하면 될 줄 알았는데, 합성 데이터로 `test.py --content`에 직접
넣어보니 `IndexError: index 22 is out of bounds for axis 1 with size 21`로 깨졌다.
원인: `preprocess/generate_dataset.py`의 `process_data()`는 **원본 CMU 31관절
BVH**(`datasets/cmu/test_bvh/127_21.bvh` 실측 확인)를 받아서 고정 인덱스
`[0,2,3,4,5,7,8,9,10,12,13,15,16,18,19,20,22,25,26,27,29]`로 21개만 골라 쓴다 —
즉 21관절로 이미 줄인 BVH를 주면 이 인덱싱 자체가 깨진다. 그래서 이 스크립트는
**31관절 원본 구조로 저장**하고, "관심 없는" 10개 관절(LHipJoint·RHipJoint·
LowerBack·Neck·LeftShoulder·RightShoulder·LeftFingerBase·RightFingerBase·
LThumb·RThumb — 전부 raw 파일에서 OFFSET이 0,0,0이라 기하학적으로 무해함을 확인)은
항등 회전으로 채워 넣는다. 학습 없는 결정론적 관절 대응 + 회전 복사라 SMPL 본체
형상(shape) 파라미터는 전혀 안 쓴다 — 라이선스가 걸린 SMPL 모델 파일 없이도
동작한다(회전값만 있으면 됨).

한계 (ponytail: 알려진 천장, 업그레이드 경로):
- SMPL spine2 회전은 버림(Spine1에는 SMPL spine3만 반영) — 등이 크게 굽는 동작에서
  오차 커짐. 필요해지면 spine1/spine2/spine3을 가중 합성.
- SMPL left/right_collar(쇄골) 회전도 버림 — 어깨를 크게 으쓱이는 동작에서 오차.
  CMU 골격 자체에 쇄골 관절이 있지만(LeftShoulder/RightShoulder) Motion Puzzle이
  안 쓰는 10개 중 하나라 채워봐야 결과에 반영 안 됨 — Motion Puzzle 사전학습
  가중치(21관절 고정)를 안 바꾸는 한 막다른 길.

사용법 (motion_puzzle conda env에서 실행 — Animation/Quaternions/BVH 모듈 재사용):
  conda run -n motion_puzzle python retarget_smpl_to_cmu.py --npz smpl_pose.npz --out out.bvh
  (smpl_pose.npz: "rotations" (F,24,3) 축각(axis-angle), "trans" (F,3) 루트 이동, "fps" 스칼라)
  그 뒤 그대로 Motion Puzzle --content 입력으로 쓸 수 있다:
  conda run -n motion_puzzle python test.py --content out.bvh --style <style.bvh> --output_dir <dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "external" / "motion_puzzle" / "motion"))
from Animation import Animation  # noqa: E402
from Quaternions import Quaternions  # noqa: E402
import BVH  # noqa: E402


# external/motion_puzzle/datasets/cmu/test_bvh/127_21.bvh 실측 확인(2026-07-23) —
# CMU 원본 31관절 계층 구조, 이름·부모·오프셋 전부 그 파일에서 그대로 추출.
RAW31_NAMES = [
    "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1", "LThumb",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase", "RightHandIndex1", "RThumb",
]

RAW31_PARENTS = np.array([
    -1, 0, 1, 2, 3, 4,
    0, 6, 7, 8, 9,
    0, 11, 12, 13, 14, 15,
    13, 17, 18, 19, 20, 21, 20,
    13, 24, 25, 26, 27, 28, 27,
])

RAW31_OFFSETS = np.array([
    [0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000],
    [1.363060, -1.794630, 0.839290], [2.448110, -6.726130, 0.000000],
    [2.562200, -7.039590, 0.000000], [0.157640, -0.433110, 2.322550],
    [0.000000, 0.000000, 0.000000],
    [-1.305520, -1.794630, 0.839290], [-2.542530, -6.985550, 0.000000],
    [-2.568260, -7.056230, 0.000000], [-0.164730, -0.452590, 2.363150],
    [0.000000, 0.000000, 0.000000],
    [0.028270, 2.035590, -0.193380], [0.056720, 2.048850, -0.042750],
    [0.000000, 0.000000, 0.000000],
    [-0.054170, 1.746240, 0.172020], [0.104070, 1.761360, -0.123970],
    [0.000000, 0.000000, 0.000000],
    [3.362410, 1.200890, -0.311210], [4.983000, 0.000000, 0.000000],
    [3.483560, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000],
    [0.715260, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000],
    [-3.136600, 1.374050, -0.404650], [-5.241900, 0.000000, 0.000000],
    [-3.444170, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000],
    [-0.622530, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000],
])

# generate_dataset.py의 process_data()가 그대로 쓰는 그 인덱스 배열 — 31개 중 이
# 21개만 "관심 있는" 관절로 골라낸다. 우리 리타겟 결과가 여기 정확히 꽂혀야 한다.
SELECTED_31_TO_21 = np.array([0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 25, 26, 27, 29])

# 참고용 — SMPL 표준 24관절 순서(SMPL 논문/공개 구현 공통 컨벤션, 라이선스 무관 정보).
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

# SELECTED_31_TO_21[i]번 raw 관절이 SMPL_JOINT_NAMES[CMU_FROM_SMPL[i]]의 회전을
# 그대로 받는다 — 직접 매핑(1:1), 위 "한계" 절 두 가지만 예외.
CMU_FROM_SMPL = np.array([0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 9, 12, 15, 16, 18, 20, 22, 17, 19, 21, 23])


def retarget(smpl_rotvecs: np.ndarray, root_trans: np.ndarray) -> Animation:
    """smpl_rotvecs: (F, 24, 3) 축각. root_trans: (F, 3). 31관절 Animation을 반환."""
    n_frames = smpl_rotvecs.shape[0]
    n_joints = len(RAW31_NAMES)

    angles = np.linalg.norm(smpl_rotvecs, axis=-1)
    smpl_quats = Quaternions.from_angle_axis(angles, smpl_rotvecs)  # (F, 24), 0벡터는 함수 내부에서 안전 처리
    mapped_quats = smpl_quats[:, CMU_FROM_SMPL]  # (F, 21)

    rotations = Quaternions.id((n_frames, n_joints))
    rotations.qs[:, SELECTED_31_TO_21] = mapped_quats.qs  # 나머지 10개는 항등 회전 유지(오프셋 0이라 무해)

    positions = np.tile(RAW31_OFFSETS[None, :, :], (n_frames, 1, 1))
    positions[:, 0] = root_trans

    orients = Quaternions.id(n_joints)
    return Animation(rotations, positions, orients, RAW31_OFFSETS, RAW31_PARENTS)


def save_bvh(smpl_rotvecs: np.ndarray, root_trans: np.ndarray, out_path: str, fps: float = 60.0) -> None:
    anim = retarget(smpl_rotvecs, root_trans)
    BVH.save(out_path, anim, names=RAW31_NAMES, frametime=1.0 / fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help='"rotations" (F,24,3), "trans" (F,3), "fps" 저장된 npz')
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = np.load(args.npz)
    save_bvh(data["rotations"], data["trans"], args.out, float(data["fps"]) if "fps" in data else 60.0)
    print(f"저장 완료: {args.out}")
