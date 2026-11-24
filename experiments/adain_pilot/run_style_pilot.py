"""Movement Style 파일럿 — 한 사람의 걷기 스타일을 다른 사람의 제자리 뛰기에 절대 AdaIN으로 입힌다.

  Pilot 1: P001 Walking 스타일 → P002 In-place Jogging 콘텐츠
  Pilot 2: P002 Walking 스타일 → P001 In-place Jogging 콘텐츠

팀 저장소 style_compare.py의 함수를 그대로 쓴다(수정 없음, main()·same_motion() 실행 없음).
AdaIN: absolute(group=None), frontal_sd=False(기본값). rebuild()는 adain() 안에서 호출된다.

순서 (프레임 범위는 59.94fps 기준으로 확정된 값):
  1. 원본 전체 npz에 mp_points() → (pts, root, fps)
  2. pts의 모든 관절과 root를 같은 프레임 범위로 자름
  3. 걷기 → channels() → stats()  = 스타일 통계
  4. adain(제자리 뛰기 pts, 스타일 통계, group=None)
  5. 생성 결과: ik(pts, 콘텐츠 주인 root × UNITS + [0, REST_HIP_Y, 0]) → bvh_points()   (same_motion()과 같은 루트 처리,
     ik()가 끝에서 R.ground()까지 한다)
  6. 스타일 주인·콘텐츠 주인의 실제 제자리 뛰기도 같은 ik() → bvh_points()
  7. 세 결과에 metrics()

결과(저장소 밖): dynamic_data/results/movement_style_pilot/pilot_<스타일>style_<콘텐츠>content/
  generated.bvh  — 생성 결과 (style_compare.save, CMU 31관절)
  metrics.json   — 세 결과의 6개 지표 원자료 + 실행 조건. 성공/실패 판정은 하지 않는다.

Motion Puzzle Animation.py가 numpy 1.26에서 umath_tests를 불러오려면 실행 시점에 환경변수가 필요하다:
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/run_style_pilot.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if os.environ.get("NUMPY_EXPERIMENTAL_DTYPE_API") != "1":
    raise SystemExit("NUMPY_EXPERIMENTAL_DTYPE_API=1 을 붙여 실행해야 한다 (docstring 참고)")

import numpy as np  # noqa: E402

WORKTREE = Path.home() / "Documents" / "dynamic-movement-style"  # 팀 코드(origin/main 7882963), 읽기 전용으로 import
DATA = Path.home() / "Documents" / "dynamic_data"
MP3D = DATA / "processed" / "mp3d"
OUT = DATA / "results" / "movement_style_pilot"

sys.path.insert(0, str(WORKTREE))
import style_compare as S  # noqa: E402

WALK = {"P001": (132, 276), "P002": (48, 216)}
JOG = {"P001": (432, 911), "P002": (300, 779)}
PILOTS = [("P001", "P002"), ("P002", "P001")]  # (스타일 주인, 콘텐츠 주인)


def load(person: str, motion: str, rng: tuple[int, int]):
    """원본 전체에 mp_points()를 먼저 적용한 뒤 자른다(전체 클립 기준 스무딩·루트 추정 유지)."""
    pts, root, fps = S.mp_points(MP3D / f"{person} {motion}.npz")
    a, b = rng
    if b > len(root):
        raise SystemExit(f"{person} {motion}: 프레임 범위 {rng}가 클립 길이 {len(root)}를 넘음")
    return {k: v[a:b] for k, v in pts.items()}, root[a:b], fps


def to_cmu(pts: dict, root: np.ndarray):
    anim = S.ik(pts, root * S.UNITS + np.array([0, S.R.REST_HIP_Y, 0]))
    return anim, S.bvh_points(anim, S.N31)


def main() -> None:
    for d in (MP3D, OUT):
        if str(d.resolve()).startswith(str(WORKTREE.resolve())):
            raise SystemExit(f"저장소 안 경로 금지: {d}")
    for style_p, content_p in PILOTS:
        walk_pts, _, walk_fps = load(style_p, "Walking", WALK[style_p])
        style_jog, style_root, _ = load(style_p, "Jogging", JOG[style_p])
        content_jog, content_root, fps = load(content_p, "Jogging", JOG[content_p])

        style_stats = S.stats(S.channels(walk_pts))
        gen_pts = S.adain(content_jog, style_stats, group=None)

        gen_anim, gen_cmu = to_cmu(gen_pts, content_root)
        _, style_cmu = to_cmu(style_jog, style_root)
        _, content_cmu = to_cmu(content_jog, content_root)

        odir = OUT / f"pilot_{style_p}style_{content_p}content"
        odir.mkdir(parents=True, exist_ok=True)
        S.save(gen_anim, odir / "generated.bvh", fps)
        rep = {
            "style": {"person": style_p, "motion": "M01 Walking", "frames": list(WALK[style_p]), "fps": walk_fps},
            "content": {"person": content_p, "motion": "M06 In-place Jogging", "frames": list(JOG[content_p]), "fps": fps},
            "style_owner_reference": {"person": style_p, "motion": "M06 In-place Jogging", "frames": list(JOG[style_p])},
            "adain": {"mode": "absolute", "group": None, "frontal_sd": False},
            "body_space": "CMU 31 (style_compare.ik -> bvh_points)",
            "metrics": {
                "generated": S.metrics(gen_cmu),
                "style_owner_real_jogging": S.metrics(style_cmu),
                "content_owner_real_jogging": S.metrics(content_cmu),
            },
        }
        (odir / "metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print("저장", odir)


if __name__ == "__main__":
    main()
