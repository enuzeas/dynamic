"""공통 걷기 콘텐츠 AdaIN 비교 데모 — 같은 콘텐츠에 P001·P002·P003의 걷기 스타일을 각각 입힌다.

팀 내부 시연용. 기존 코드는 수정하지 않고 `style_compare.py`와 `run_style_pilot.load()`를 그대로 쓴다.

콘텐츠 (세 참가자 모두 동일):
  공통 걷기 모캡 BVH(별도 확보, 출처·라이선스 미확인, 저장소에 포함하지 않음), 프레임 [9020, 9200) = 180프레임, 60fps
  - Motion Puzzle `BVH.load(path, start, end)`는 start <= i < end-1 만 읽으므로 end=9201을 넘긴다(전체 로드와 일치 검증).
  - 단위: 이 BVH는 cm. 루트는 `m × UNITS_PER_M`(기존 MediaPipe 경로의 관례 `root * UNITS`)로 CMU 단위에 맞춘다.
    높이는 `ik()` 끝의 `R.ground()`가 상수로 맞추므로 REST_HIP_Y는 더하지 않는다. 수평은 첫 프레임을 원점으로 옮긴다.
스타일 (기존 NPZ, 영상 자체는 열지 않음):
  P001 `P001 Walking-M01.npz` [30, 210), P002 `P002 Walking.npz` [48, 216), P003 `P003 Walking.npz` [228, 408)
적용: 기존 절대 AdaIN (`group=None`, `frontal_sd=False`) → `ik()` → `save()`. 강도 조절은 없다.

출력은 새 디렉터리 `dynamic_data/results/common_walk_demo_adain/`에만 쓰고, 이미 있으면 중단한다.

실행:
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/run_common_walk_demo.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

if os.environ.get("NUMPY_EXPERIMENTAL_DTYPE_API") != "1":
    raise SystemExit("NUMPY_EXPERIMENTAL_DTYPE_API=1 을 붙여 실행해야 한다 (docstring 참고)")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_style_pilot as X  # noqa: E402  (style_compare를 X.S로 불러온다)

S = X.S
CONTENT = Path("공통_걷기.bvh")  # 공통 걷기 모캡 BVH(별도 확보)의 실제 경로로 바꿔 쓴다
WINDOW = (9020, 9200)
OUT = X.DATA / "results" / "common_walk_demo_adain"
STYLES = [  # (이름, 참가자 라벨, NPZ 모션 라벨, 프레임 구간)
    ("P001", "P001", "Walking-M01", (30, 210)),
    ("P002", "P002", "Walking", (48, 216)),
    ("P003", "P003", "Walking", (228, 408)),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    if str(OUT.resolve()).startswith(str(X.WORKTREE.resolve())):
        raise SystemExit(f"저장소 안 경로 금지: {OUT}")
    if OUT.exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {OUT}")
    for p in (CONTENT, *(X.MP3D / f"{a} {b}.npz" for _, a, b, _ in STYLES)):
        if not p.exists():
            raise SystemExit(f"입력 없음: {p}")

    # 공통 콘텐츠: 창 읽기 → 전체 읽기와 일치 검증
    a, b = WINDOW
    canim, cnames, cft = S.BVH.load(str(CONTENT), a, b + 1)
    full, fnames, _ = S.BVH.load(str(CONTENT))
    assert len(canim.rotations) == b - a, f"콘텐츠 프레임 수 {len(canim.rotations)} != {b - a}"
    assert np.allclose(canim.positions, full.positions[a:b]) and np.allclose(canim.rotations.qs, full.rotations.qs[a:b]), "창 읽기가 전체 슬라이스와 다름"
    cpts = S.bvh_points(canim, cnames)
    root_m = (canim.positions[:, 0] - canim.positions[0, 0] * np.array([1, 0, 1])) / 100.0  # cm → m, 수평 원점 이동
    croot = root_m * S.UNITS
    fps_out = 1.0 / cft

    OUT.mkdir(parents=True, exist_ok=False)
    report = {"content": {"file": str(CONTENT), "frames": [a, b], "n_frames": b - a, "fps": fps_out, "metrics": S.metrics(cpts)}, "styles": {}}

    # 기준선: 같은 IK 경로를 거친 콘텐츠 (스타일 없음)
    ref = S.ik(cpts, croot)
    S.save(ref, OUT / "content_original_ik.bvh", fps_out)

    for name, person, motion, rng in STYLES:
        spts, _, sfps = X.load(person, motion, rng)
        stats = S.stats(S.channels(spts))
        gen = S.adain(cpts, stats, group=None)  # frontal_sd 기본값(False)
        anim = S.ik(gen, croot)
        dst = OUT / f"{name}_style_commonwalk.bvh"
        S.save(anim, dst, fps_out)
        report["styles"][name] = {"npz": f"{person} {motion}.npz", "frames": list(rng), "n_style_frames": rng[1] - rng[0], "style_fps": sfps,
                                  "output": str(dst), "metrics": S.metrics(S.bvh_points(anim, S.N31))}

    # 저장 결과 다시 읽어 프레임 수 확인
    for f in sorted(OUT.glob("*.bvh")):
        an, nm, ft = S.BVH.load(str(f))
        report.setdefault("saved", {})[f.name] = {"frames": len(an.rotations), "joints": len(nm), "frame_time": ft}
    report["adain"] = {"mode": "absolute", "group": None, "frontal_sd": False}
    report["units"] = {"content_unit": "cm (BVH)", "root_to_cmu": "root_m * UNITS_PER_M", "UNITS_PER_M": float(S.UNITS)}
    report["env"] = {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable,
                     "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    report["scripts_sha256"] = {p.name: sha256(p) for p in (Path(__file__), HERE / "run_style_pilot.py", X.WORKTREE / "style_compare.py")}
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for k, v in report["saved"].items():
        print(f"{k}: {v['frames']} frames, {v['joints']} joints, frame_time {v['frame_time']:.6f}")
    print("저장", OUT)


if __name__ == "__main__":
    main()
