"""sample/A·B·C 반복 촬영 3회씩에서 동작 서명 스칼라를 뽑아 evaluate_icc.py용 npz로 저장.

서명 추출은 src/analyze.py·src/dynamic_id.py와 같은 경로(pose_extract.extract_joint_dynamics
+ demo.trim_motion, JOINT="오른손목")를 그대로 쓴다. 다만 그 둘은 트림된 속도 곡선 전체를
DTW로 비교하는 반면, evaluate_icc.py의 ICC(3,1)은 사람×반복 셀 하나에 스칼라 값 하나가
필요해서 트림된 속도 곡선의 평균값 하나로 요약한다.

sample/original/엔 A·B만 있고 C가 없어서(확인함) 원본 화질인 sample/ 루트의 C_*.MOV로 대체.

사용법 (repo 루트에서, mediapipe 설치된 venv로):
  .venv/bin/python build_icc_features.py --out icc_features.npz
  .venv/bin/python evaluate_icc.py --npz icc_features.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import extract_joint_dynamics, LANDMARK_INDEX  # noqa: E402
from demo import trim_motion  # noqa: E402

JOINT = "오른손목"  # analyze.py·dynamic_id.py와 동일한 단일 관절(기본값)
ALL_JOINTS = tuple(LANDMARK_INDEX)  # scope.md MVP 스펙의 다관절 결합(8개: 양팔+양쪽 골반)

# 관절 프리셋 — 어떤 관절들을 평균으로 묶어 서명을 낼지. multi(8개 전부)는 M02("손 뻗기")처럼
# 한쪽 팔만 쓰는 비대칭 동작에서 오히려 신호를 희석시켰다(project_track1_icc_pilot_result
# 메모 참고) — right_arm처럼 실제로 그 동작에 관여하는 관절만 묶는 프리셋을 추가해가는 구조.
JOINT_PRESETS = {
    "single": (JOINT,),
    "multi": ALL_JOINTS,
    "right_arm": ("오른손목", "오른팔꿈치", "오른어깨"),
    "hips": ("오른엉덩이", "왼엉덩이"),  # M04("앉았다 일어서기")의 권장 신호(무게중심·골반 정렬)
    "hands": ("오른손목", "왼손목"),  # M03("설명하며 말하기")의 권장 신호(무의식적 손 제스처)
}

SUBJECTS = {
    "A": ["sample/original/A_1.mp4", "sample/original/A_2.mp4", "sample/original/A_3.mp4"],
    "B": ["sample/original/B_1.mp4", "sample/original/B_2.mp4", "sample/original/B_3.mp4"],
    "C": ["sample/C_1.MOV", "sample/C_2.MOV", "sample/C_3.MOV"],
}


def _trim_window(speed: np.ndarray, threshold: float = 0.15, pad: int = 3) -> slice:
    """demo.trim_motion과 같은 구간을 슬라이스로 반환 — jerk 등 다른 시계열에도 같은
    구간을 적용하기 위해서다. trim_motion은 넘겨받은 시계열 자신의 피크 기준으로 자르므로,
    jerk에 그대로 다시 부르면 speed와는 다른(자기 자신 기준) 구간이 나온다."""
    moving = np.flatnonzero(speed > threshold * speed.max())
    if len(moving) == 0:
        return slice(0, len(speed))
    start = max(0, moving[0] - pad)
    end = min(len(speed), moving[-1] + pad + 1)
    return slice(start, end)


def _joint_scalar(dyn_joint: dict, feature: str) -> float:
    if feature == "mean_speed":
        return float(np.mean(trim_motion(dyn_joint["speed"])))
    if feature == "mean_jerk":
        return float(np.mean(dyn_joint["jerk"][_trim_window(dyn_joint["speed"])]))
    raise ValueError(f"알 수 없는 feature: {feature!r}")


def signature(video_path: str, feature: str = "mean_speed", joints: tuple[str, ...] = (JOINT,)) -> float:
    """영상 하나 → 관절별 스칼라 서명의 평균.

    joints 기본값은 analyze.py·dynamic_id.py와 같은 단일 관절(오른손목) — 관절 하나만 주면
    기존과 동일하게 그 관절 값 그대로다. ALL_JOINTS(8개)를 주면 scope.md MVP 스펙의 다관절
    결합이 된다 — 관절 하나의 우연한 노이즈가 평균으로 상쇄되길 기대하는 것.

    feature="mean_speed"(기본, 트림된 속도 곡선 평균) 또는 "mean_jerk" — 후자는
    `쌤쌤_촬영_체크리스트_v2.md` §4가 M02("손 뻗기")의 권장 추출 신호로 지목한 저크를,
    speed 기준으로 잡은 같은 트림 구간에 적용한 평균값이다.
    """
    dyn = extract_joint_dynamics(video_path, joints)
    return float(np.mean([_joint_scalar(dyn[j], feature) for j in joints]))


def build_features(
    subjects: dict[str, list[str]], extract_fn: Callable[[str], float] = signature
) -> tuple[list[str], np.ndarray]:
    """subjects(사람 -> 반복별 파일 목록) -> (사람 순서, (n_subjects, k_repeats) 배열)."""
    names = list(subjects)
    features = np.array([[extract_fn(v) for v in subjects[name]] for name in names])
    return names, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", choices=["mean_speed", "mean_jerk"], default="mean_speed")
    parser.add_argument("--joints", choices=list(JOINT_PRESETS), default="single")
    parser.add_argument("--out", default=None, help="기본: icc_features_{feature}_{joints}.npz")
    args = parser.parse_args()
    joints = JOINT_PRESETS[args.joints]
    out = args.out or f"icc_features_{args.feature}_{args.joints}.npz"

    for name, files in SUBJECTS.items():
        for f in files:
            if not Path(f).is_file():
                raise SystemExit(f"파일 없음: {f}")

    print(f"서명 추출 중 ({sum(len(v) for v in SUBJECTS.values())}개 영상, feature={args.feature}, joints={args.joints})...")
    names, features = build_features(SUBJECTS, extract_fn=lambda p: signature(p, args.feature, joints))
    for name, row in zip(names, features):
        print(f"  {name}: {['%.4f' % v for v in row]}")

    np.savez(out, features=features, subjects=np.array(names))
    print(f"\n저장 완료: {out}  (shape={features.shape})")
