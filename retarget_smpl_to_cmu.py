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
  (smpl_pose.npz: "rotations" (F,24,3) 축각(axis-angle), "trans" (F,3) 루트 이동(미터), "fps" 스칼라)
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

# 루트 이동(미터) -> CMU 단위 변환. 2026-09-11에 실측으로 잡은 값이다.
# CMU BVH는 cm가 아니라 자체 단위다: 이 골격의 대퇴 |offs[3]|=7.16 + 경골 |offs[4]|=7.49 = 14.65단위가
# SMPL 템플릿 다리 길이 약 0.82 m에 해당하므로 미터당 약 17.9단위. 예전엔 ×100(cm)을 써서 캐릭터가
# 바닥 아래까지 내려가고 상하로 6배 튀었다.
SMPL_LEG_M = 0.82
UNITS_PER_M = (np.linalg.norm(RAW31_OFFSETS[3]) + np.linalg.norm(RAW31_OFFSETS[4])) / SMPL_LEG_M

# GVHMR transl은 절대 높이가 아니라 첫 프레임 기준 상대 변위다(실측: 평균 0.06 m, 음수도 나옴).
# 그래서 골격이 바닥에 서는 높이를 더해야 한다. 값은 아래 REST_ALIGN 정의 뒤에서 FK로 계산한다
# — 쉴 때 자세 보정을 하면 다리가 덜 벌어져 발이 더 내려가므로 오프셋 단순 합으로는 안 맞는다.

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


# --- 쉴 때 자세 보정 (2026-09-11 추가) ---------------------------------------
# 회전만 복사하면 안 되는 이유: CMU 골격은 쉴 때부터 다리를 각각 수직에서 20° 벌리고 어깨를 20°
# 올린 자세다. SMPL 템플릿은 대퇴가 수직에서 5.2°, 상완이 수평에서 2.8°다. SMPL 회전이 0일 때
# CMU 골격은 자기 쉴 때 자세로 가므로, 보정 없이는 양다리가 30° 과하게 벌어진 채로 달린다
# (2026-09-11 실제 산출물에서 눈으로 확인). CMU 원본 모션이 멀쩡한 건 그 데이터가 이 골격용으로
# 만들어져 회전값에 안쪽으로 접는 보정이 이미 들어 있기 때문이다.
#
# 보정: 관절 j가 회전시키는 뼈의 CMU 쉴때 방향을 SMPL 쉴때 방향으로 돌리는 쿼터니언 A_j를 두고
#   q_cmu_local[j] = A_parent⁻¹ ∘ q_smpl_local[j] ∘ A_j
# 를 쓴다. 그러면 SMPL 회전이 0일 때 CMU 골격이 SMPL 쉴 때 자세를 재현한다.
#
# 아래 값은 SMPL_NEUTRAL.pkl의 템플릿 관절 위치(J)에서 계산한 단위 방향 벡터다. 상수로 구워
# 넣어서 런타임에는 여전히 SMPL 모델 파일이 필요 없다(라이선스). 유도 방법은
# doc/samsam/smpl_to_bvh_calibration.md 참고.
#
# Neck1(15)은 일부러 뺐다 — SMPL의 head 관절은 목보다 앞쪽에 있고(방향 Z +0.62) CMU Head는
# 거의 수직 위라 관절 의미가 다르다. 보정하면 머리가 42° 앞으로 숙여진다. 의미가 다른 건
# 보정이 아니라 왜곡이므로 그대로 둔다.
SMPL_REST_DIR = {
     1: ( 0.604302, -0.794550, -0.059242),   # LHipJoint     보정 23.8도
     2: ( 0.090970, -0.995782, -0.011932),   # LeftUpLeg     보정 14.8도
     3: (-0.033940, -0.993454, -0.109074),   # LeftLeg       보정 22.8도
     4: ( 0.196260, -0.415413,  0.888208),   # LeftFoot      보정 16.2도
     6: (-0.598312, -0.800352, -0.038194),   # RHipJoint     보정 22.9도
     7: (-0.099563, -0.994765, -0.023012),   # RightUpLeg    보정 14.3도
     8: ( 0.039339, -0.993638, -0.105525),   # RightLeg      보정 23.0도
     9: (-0.188191, -0.357099,  0.914912),   # RightFoot     보정 12.5도
    11: (-0.022572,  0.971026, -0.237904),   # LowerBack     보정  8.6도
    12: ( 0.036530,  0.989548,  0.139498),   # Spine         보정  9.2도
    13: (-0.012737,  0.980428, -0.196466),   # Spine1        보정 17.0도
    17: ( 0.944210,  0.316223, -0.092039),   # LeftShoulder  보정  1.2도
    18: ( 0.993266, -0.048865, -0.105047),   # LeftArm       보정  6.7도
    19: ( 0.999340,  0.036031, -0.004695),   # LeftForeArm   보정  2.1도
    20: ( 0.980084, -0.095178, -0.174290),   # LeftHand      보정 11.5도
    24: (-0.943224,  0.319782, -0.089821),   # RightShoulder 보정  5.2도
    25: (-0.995100, -0.052273, -0.083928),   # RightArm      보정  5.7도
    26: (-0.999300,  0.030423, -0.021760),   # RightForeArm  보정  2.1도
    27: (-0.990107, -0.071574, -0.120690),   # RightHand     보정  8.1도
}


def _cmu_bone_dir(j: int) -> np.ndarray:
    """관절 j가 회전시키는 뼈의 CMU 쉴때 방향(단위벡터). 자식 오프셋이 0이면 그 아래로 누적."""
    kids = [k for k, p in enumerate(RAW31_PARENTS) if p == j]
    for k in kids:
        v, cur = RAW31_OFFSETS[k].copy(), k
        while np.allclose(v, 0):
            nxt = [m for m, p in enumerate(RAW31_PARENTS) if p == cur]
            if not nxt:
                break
            cur = nxt[0]
            v = v + RAW31_OFFSETS[cur]
        if not np.allclose(v, 0):
            return v / np.linalg.norm(v)
    return None


def _rest_align() -> Quaternions:
    """관절별 A_j (31개). 보정 대상이 아니면 항등."""
    A = Quaternions.id(len(RAW31_NAMES))
    for j, smpl_dir in SMPL_REST_DIR.items():
        cmu_dir = _cmu_bone_dir(j)
        if cmu_dir is None:
            continue
        A.qs[j] = Quaternions.between(cmu_dir[None, :], np.array([smpl_dir]))[0].qs[0]
    return A


REST_ALIGN = _rest_align()


def _rest_hip_height() -> float:
    """보정된 쉴 때 자세(=SMPL 쉴 때 자세)에서 가장 낮은 관절이 바닥(y=0)에 닿는 힙 높이."""
    q = Quaternions.id(len(RAW31_NAMES))
    q = (-Quaternions(np.concatenate([Quaternions.id(1).qs, REST_ALIGN.qs[RAW31_PARENTS[1:]]]))) * q * REST_ALIGN
    gq, gp = [], []
    for j, parent in enumerate(RAW31_PARENTS):
        if parent < 0:
            gq.append(q[j:j + 1]); gp.append(np.zeros(3))
        else:
            gq.append(gq[parent] * q[j:j + 1])
            gp.append(gp[parent] + (gq[parent] * RAW31_OFFSETS[j][None, :])[0])
    return -float(min(p[1] for p in gp))


# 보정 후 실측 16.4 (보정 전 오프셋 합 15.99). CMU 참조 41_02.bvh의 루트 Y 최소 16.00과 같은 자릿수.
REST_HIP_Y = _rest_hip_height()


def detrend_height(trans: np.ndarray) -> np.ndarray:
    """루트 높이의 선형 추세를 제거한다(평균 높이는 유지). trans는 (F,3) 미터.

    2026-09-11 실측: `-s`(정적 카메라, SLAM 생략)로 뽑은 GVHMR 출력에서 5.91초 동안 루트가
    0.17 m 꾸준히 올라갔다 — 수평 1.81 m 이동 대비 실효 경사 5.4도. 촬영 가이드상 바닥은
    평평하므로(`samsam_shooting_guide.md`) 이 추세는 월드 높이 추정 드리프트다. 추세를 빼도
    뛰는 상하 진폭(잔차 ±1.3단위)은 그대로 남는다.

    ponytail: 선형 추세만 뺀다. 클립 전체에서 실제로 높이가 변하는 동작(계단, 앉은 채로 끝나는
    촬영)에는 맞지 않으므로 `--keep-height-drift`로 끌 수 있게 뒀다. 곡률까지 남으면 그때
    저역통과 필터로 올려도 된다.
    """
    y = trans[:, 1]
    t = np.arange(len(y), dtype=float)
    slope = np.polyfit(t, y, 1)[0]
    out = trans.copy()
    out[:, 1] = y - slope * t
    return out


def retarget(smpl_rotvecs: np.ndarray, root_trans: np.ndarray) -> Animation:
    """smpl_rotvecs: (F, 24, 3) 축각. root_trans: (F, 3) **미터**. 31관절 Animation을 반환."""
    n_frames = smpl_rotvecs.shape[0]
    n_joints = len(RAW31_NAMES)

    angles = np.linalg.norm(smpl_rotvecs, axis=-1)
    smpl_quats = Quaternions.from_angle_axis(angles, smpl_rotvecs)  # (F, 24), 0벡터는 함수 내부에서 안전 처리
    mapped_quats = smpl_quats[:, CMU_FROM_SMPL]  # (F, 21)

    rotations = Quaternions.id((n_frames, n_joints))
    rotations.qs[:, SELECTED_31_TO_21] = mapped_quats.qs  # 나머지 10개는 SMPL 회전 없음(항등)

    # 쉴 때 자세 보정: q_cmu_local[j] = A_parent⁻¹ ∘ q_smpl_local[j] ∘ A_j
    # 보정 대상이 아닌 관절은 A가 항등이라 이 식이 회전 복사와 같아진다.
    A = REST_ALIGN
    A_parent = Quaternions(np.concatenate([Quaternions.id(1).qs, A.qs[RAW31_PARENTS[1:]]]))
    rotations = (-A_parent)[None] * rotations * A[None]

    positions = np.tile(RAW31_OFFSETS[None, :, :], (n_frames, 1, 1))
    positions[:, 0] = root_trans * UNITS_PER_M
    positions[:, 0, 1] += REST_HIP_Y

    orients = Quaternions.id(n_joints)
    return Animation(rotations, positions, orients, RAW31_OFFSETS, RAW31_PARENTS)


def save_bvh(smpl_rotvecs: np.ndarray, root_trans: np.ndarray, out_path: str, fps: float = 60.0,
             keep_height_drift: bool = False) -> None:
    if not keep_height_drift:
        root_trans = detrend_height(root_trans)
    anim = retarget(smpl_rotvecs, root_trans)
    BVH.save(out_path, anim, names=RAW31_NAMES, frametime=1.0 / fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help='"rotations" (F,24,3), "trans" (F,3), "fps" 저장된 npz')
    parser.add_argument("--keep-height-drift", action="store_true",
                        help="루트 높이의 선형 추세를 남긴다(기본은 제거 — 바닥이 평평하다는 전제)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = np.load(args.npz)
    save_bvh(data["rotations"], data["trans"], args.out, float(data["fps"]) if "fps" in data else 60.0,
             keep_height_drift=args.keep_height_drift)
    print(f"저장 완료: {args.out}")
