"""retarget_smpl_to_cmu 자체 검증 — SMPL 라이선스 모델 없이 합성 축각 데이터로
(1) BVH 저장·재로드 왕복 구조가 맞는지, (2) Motion Puzzle `test.py --content`에
실제로 먹히는지(제일 중요한 통합 검증 — 21관절로 줄여서 저장했다가 이 검증에서
막혀 31관절 구조로 다시 짠 이력이 있다)까지 확인한다.

실행 (motion_puzzle conda env 필요 — Animation/Quaternions/BVH·test.py 재사용 때문):
  conda run -n motion_puzzle python test_retarget_smpl_to_cmu.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from retarget_smpl_to_cmu import RAW31_NAMES, RAW31_PARENTS, save_bvh

REPO_ROOT = Path(__file__).resolve().parent
MOTION_PUZZLE_DIR = REPO_ROOT / "external" / "motion_puzzle"
sys.path.insert(0, str(MOTION_PUZZLE_DIR / "motion"))
import BVH  # noqa: E402


def make_synthetic_smpl_motion(n_frames: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """SMPL 본체(shape) 없이 회전만 합성 — right_elbow(19)만 앞뒤로 굽히고 나머진 정지."""
    rotvecs = np.zeros((n_frames, 24, 3))
    t = np.linspace(0, 2 * np.pi, n_frames)
    rotvecs[:, 19, 2] = 0.8 * np.sin(t)  # right_elbow를 z축으로 굽힘
    trans = np.zeros((n_frames, 3))
    trans[:, 1] = 100.0  # 대략적인 힙 높이
    trans[:, 2] = np.linspace(0, 20, n_frames)  # 앞으로 약간 걸어나감
    return rotvecs, trans


def check_roundtrip(bvh_path: str) -> None:
    anim, names, _ = BVH.load(bvh_path)
    assert names == RAW31_NAMES, "저장 후 다시 읽은 관절 이름이 원래 목록과 달라졌다"
    assert anim.shape[1] == len(RAW31_NAMES) == 31
    assert anim.shape[0] == 60
    assert np.array_equal(anim.parents, RAW31_PARENTS), "저장 후 부모 인덱스가 달라졌다"
    assert not np.isnan(anim.rotations.qs).any(), "회전값에 NaN이 있다"
    print("  왕복 검증 통과 (31관절, 60프레임, NaN 없음)")


def check_motion_puzzle_integration(bvh_path: str) -> None:
    """진짜 통합 검증 — Motion Puzzle test.py에 --content로 직접 먹이기."""
    style_bvh = MOTION_PUZZLE_DIR / "datasets" / "cmu" / "test_bvh" / "142_21.bvh"
    out_dir = MOTION_PUZZLE_DIR / "output" / "dev_retarget_selfcheck"
    result = subprocess.run(
        [sys.executable, "test.py", "--content", bvh_path, "--style", str(style_bvh), "--output_dir", str(out_dir)],
        cwd=str(MOTION_PUZZLE_DIR), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"test.py가 우리 리타겟 BVH를 못 받아들였다:\n{result.stderr[-2000:]}"

    produced = list(out_dir.glob("Style_*_Content_*_fixed.bvh"))
    assert produced, f"스타일 전이 출력 BVH가 안 생겼다: {list(out_dir.iterdir())}"
    print(f"  Motion Puzzle 통합 검증 통과 — 출력: {produced[0].name}")


def main() -> None:
    out_path = "/tmp/retarget_test_out.bvh"
    rotvecs, trans = make_synthetic_smpl_motion()
    save_bvh(rotvecs, trans, out_path, fps=60.0)
    print(f"합성 리타겟 BVH 저장: {out_path}")

    check_roundtrip(out_path)
    check_motion_puzzle_integration(out_path)

    print("자체 검증 통과")


if __name__ == "__main__":
    main()
