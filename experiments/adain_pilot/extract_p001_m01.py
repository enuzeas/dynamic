"""P001 M01 자연 걷기 — MediaPipe 3D 추출 (기존 `P001 Walking.npz`(M07)와 분리 저장).

기존 `extract_pilot.py`는 수정하지 않는다. 팀 저장소의 `mp3d_extract.extract(path)`만 그대로 쓰고,
입력은 참가자 영상 폴더의 원본을 읽기만 한다. 출력은 기존 NPZ와 다른 이름으로 저장한다.

  입력: ~/work/stymo_trial/participant_data/video/P001_S1_M01__T01_CAM_F.mov
  출력: ~/Documents/dynamic_data/processed/mp3d/P001 Walking-M01.npz
  구간: [30, 210) — 0-based, 끝 제외, 180프레임 (사용자 확정 0.5~3.5초). NPZ에는 영상 전체를 저장하고
        구간은 로더에서 `load("P001", "Walking-M01", (30, 210))`처럼 자른다(기존 NPZ와 같은 방식).

출력 파일이 이미 있으면 덮어쓰지 않고 중단한다(배타적 생성 모드). 저장소 안 경로는 거부한다.

실행 (기존 파일럿과 같은 환경):
  /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/extract_p001_m01.py
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path.home() / "Documents" / "dynamic-movement-style"  # 팀 코드, 읽기 전용으로 import
SRC = Path.home() / "work" / "stymo_trial" / "participant_data" / "video" / "P001_S1_M01__T01_CAM_F.mov"
DST = Path.home() / "Documents" / "dynamic_data" / "processed" / "mp3d" / "P001 Walking-M01.npz"
WINDOW = (30, 210)

sys.path.insert(0, str(WORKTREE))
from mp3d_extract import extract  # noqa: E402


def main() -> None:
    if str(DST.resolve()).startswith(str(WORKTREE.resolve())):
        raise SystemExit(f"저장소 안 경로 금지: {DST}")
    if not SRC.exists():
        raise SystemExit(f"원본 없음: {SRC}")
    if DST.exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {DST}")

    import cv2
    import mediapipe
    import pose_extract  # mp3d_extract가 src/를 sys.path에 넣어 둔다

    d = extract(SRC)
    model = Path(pose_extract.MODEL_PATH)
    meta = {
        "source_name": SRC.name,
        "source_size": SRC.stat().st_size,
        "source_path": str(SRC),
        "window_note": f"intended window [{WINDOW[0]}, {WINDOW[1]}) 0-based end-exclusive; NPZ holds the whole clip",
        "env_python": platform.python_version(),
        "env_numpy": np.__version__,
        "env_mediapipe": getattr(mediapipe, "__version__", "unknown"),
        "env_opencv": cv2.__version__,
        "pose_model": model.name,
        "pose_model_size": model.stat().st_size,
        "extractor": "dynamic-movement-style/mp3d_extract.extract",
        "extractor_script": Path(__file__).name,
    }
    with open(DST, "xb") as f:  # 'x' = 파일이 있으면 실패, 덮어쓰지 않음
        np.savez(f, **d, **meta)
    print("저장", DST, d["world"].shape, d["xy"].shape, f"{d['fps']:.5f}fps")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
