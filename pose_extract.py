"""BIO-IP 관절 추출 — MediaPipe PoseLandmarker(Tasks API) 기반.

demo.py가 쓰던 화면 전체 옵티컬 플로우 placeholder를 대체한다.
관절별 위치를 평활화한 뒤 미분해 속도·가속도·저크를 구한다 — 평활화 없이
두 번, 세 번 미분하면 포즈 추정 노이즈가 그대로 증폭되기 때문이다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.signal import savgol_filter

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

# MediaPipe Pose 33 랜드마크 인덱스 중 다이내믹스 서명이 잘 드러나는 오른팔 관절
LANDMARK_INDEX = {
    "오른손목": 16,
    "오른팔꿈치": 14,
    "오른어깨": 12,
}

_SMOOTH_WINDOW = 7  # ponytail: 고정 윈도우. fps·촬영거리 편차가 크면 조정 필요
_SMOOTH_POLY = 3


def _make_landmarker() -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _smooth(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    window = min(_SMOOTH_WINDOW, n - (1 - n % 2))
    if window < _SMOOTH_POLY + 2:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=_SMOOTH_POLY, axis=0)


def extract_joint_dynamics(
    video_path: str, joints: tuple[str, ...] = tuple(LANDMARK_INDEX)
) -> dict[str, dict[str, np.ndarray]]:
    """영상 → 관절별 {speed, accel, jerk} 시계열."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

    landmarker = _make_landmarker()
    positions: dict[str, list[tuple[float, float]]] = {j: [] for j in joints}
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_i / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            for j in joints:
                if landmarks is not None:
                    p = landmarks[LANDMARK_INDEX[j]]
                    positions[j].append((p.x * w, p.y * h))
                elif positions[j]:
                    positions[j].append(positions[j][-1])  # 검출 실패 프레임은 직전 위치 유지
                else:
                    positions[j].append((0.0, 0.0))
            frame_i += 1
    finally:
        cap.release()
        landmarker.close()

    if frame_i < 10:
        raise ValueError(f"동작 추출 실패 (프레임 부족): {video_path}")

    dynamics = {}
    for j in joints:
        pos = _smooth(np.array(positions[j], dtype=float))
        vel = np.gradient(pos, axis=0) * fps
        speed = _smooth(np.linalg.norm(vel, axis=1))
        accel = np.gradient(speed) * fps
        jerk = np.abs(np.gradient(accel) * fps)
        dynamics[j] = {"speed": speed, "accel": accel, "jerk": jerk}
    return dynamics
