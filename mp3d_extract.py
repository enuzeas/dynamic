"""MediaPipe 3D(world landmarks) 추출 — 스타일 전이(Motion Puzzle·관절 각도 오프셋) 실험용.

GVHMR(GPU·Colab)을 거치지 않고 로컬 CPU로 참가자 3D 동작을 빠르게 얻기 위한 근사 경로다.
MediaPipe world landmarks: 골반 중심 원점, 미터 단위, 카메라 축(x 오른쪽, y 아래, z 깊이).
루트(골반)의 이동 경로는 world 좌표에 없어서 2D 영상 위치와 몸통 길이 변화(원근)로 따로 추정한다.

출력: reports/mp3d/{클립 이름}.npz — world (T,33,3), xy (T,33,2, 프레임 대각선 정규화), fps
사용법: .venv/bin/python mp3d_extract.py P001_S1_M06_T01 P001_S1_M01_T01 ...   (이름만, data/raw/_nosound에서 찾음)
       .venv/bin/python mp3d_extract.py --set                                     # 비교 실험에 쓰는 기본 묶음
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import _make_landmarker  # noqa: E402

RAW = Path("data/raw/_nosound")
OUT = Path("reports/mp3d")
PEOPLE = "P001,P002,P003,P004,P006,P007,P008,P010,P011,P012,P013,P014".split(",")


def extract(path: Path) -> dict:
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    diag = (w ** 2 + h ** 2) ** 0.5
    lm = _make_landmarker()
    world, xy, i = [], [], 0
    lw, lx = np.zeros((33, 3)), np.zeros((33, 2))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                    int(i * 1000 / fps))
            if r.pose_world_landmarks:
                lw = np.array([(p.x, p.y, p.z) for p in r.pose_world_landmarks[0]])
                lx = np.array([(p.x * w / diag, p.y * h / diag) for p in r.pose_landmarks[0]])
            world.append(lw)
            xy.append(lx)
            i += 1
    finally:
        cap.release()
        lm.close()
    return {"world": np.array(world), "xy": np.array(xy), "fps": fps}


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = OUT / f"{n}.npz"
        if dst.exists():
            continue
        d = extract(RAW / f"{n}_CAM_F.mov")
        np.savez(dst, **d)
        print("저장", dst, d["world"].shape)


if __name__ == "__main__":
    if "--set" in sys.argv:
        names = []
        for p in PEOPLE:
            names.append(f"{p}_S1_M06_T01")  # 스타일 원천: 조깅
            names += [f.name.replace("_CAM_F.mov", "") for f in sorted(RAW.glob(f"{p}_S1_M01_*T*_CAM_F.mov"))]  # 실제 걷기(정답 비교용)
        main(names)
    else:
        main(sys.argv[1:])
