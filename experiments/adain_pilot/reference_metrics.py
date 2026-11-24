"""Movement Style batch9 — 실제 데이터 6개(3명 × 걷기·제자리 뛰기)의 기준 지표를 CMU 31 몸 공간에서 계산한다.

생성 결과와 완전히 같은 경로를 쓰려고 run_style_pilot의 load()(전체 npz에 mp_points 후 자르기)와
to_cmu()(ik -> bvh_points, same_motion과 같은 루트 처리)를 그대로 import한다. 기존 파일은 수정하지 않는다.

출력: dynamic_data/results/movement_style_batch9_absolute/references.json (이미 있으면 덮어쓰지 않고 멈춘다)

실행:
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/reference_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_style_pilot as X  # noqa: E402  (NUMPY_EXPERIMENTAL_DTYPE_API=1 검사도 여기서 한다)

OUT = X.DATA / "results" / "movement_style_batch9_absolute"
RANGES = {  # 확정 구간 (59.94fps 프레임, [start, end))
    "P001": {"Walking": (132, 276), "Jogging": (432, 911)},
    "P002": {"Walking": (48, 216), "Jogging": (300, 779)},
    "P003": {"Walking": (228, 408), "Jogging": (150, 629)},
}
MOTION_NAME = {"Walking": "M01 Walking", "Jogging": "M06 In-place Jogging"}


def main() -> None:
    dst = OUT / "references.json"
    if str(OUT.resolve()).startswith(str(X.WORKTREE.resolve())):
        raise SystemExit(f"저장소 안 경로 금지: {OUT}")
    if dst.exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {dst}")
    refs = {}
    for person, motions in RANGES.items():
        for motion, rng in motions.items():
            pts, root, fps = X.load(person, motion, rng)
            _, cmu = X.to_cmu(pts, root)
            refs[f"{person} {motion}"] = {
                "person": person, "motion": MOTION_NAME[motion], "frames": list(rng), "n_frames": rng[1] - rng[0],
                "fps": fps, "metrics": X.S.metrics(cmu),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"body_space": "CMU 31 (style_compare.ik -> bvh_points)", "references": refs},
                              ensure_ascii=False, indent=1))
    print("저장", dst)


if __name__ == "__main__":
    main()
