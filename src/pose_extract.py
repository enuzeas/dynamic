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
HAND_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# MediaPipe Pose 33 랜드마크 인덱스 중 다이내믹스 서명이 잘 드러나는 관절 —
# scope.md MVP 스펙(6~8개, 다관절 결합)에 맞춰 양팔 + 양쪽 골반으로 확장.
# dynamics_layer.md의 "댄스·안무 핵심 관절: 손목·팔꿈치·어깨·골반"과 일치.
LANDMARK_INDEX = {
    "오른손목": 16,
    "오른팔꿈치": 14,
    "오른어깨": 12,
    "왼손목": 15,
    "왼팔꿈치": 13,
    "왼어깨": 11,
    "오른엉덩이": 24,
    "왼엉덩이": 23,
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
    # 프레임 대각선으로 정규화 — 해상도가 다른 영상끼리 픽셀 속도를 비교 가능하게 만든다.
    # (예: 3840x2160 영상은 1080x1920 영상보다 같은 동작에서도 픽셀 이동량이 커진다)
    diag = (w ** 2 + h ** 2) ** 0.5

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
                    positions[j].append((p.x * w / diag, p.y * h / diag))
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


INDEX_FINGERTIP = 8  # MediaPipe Hand의 21개 랜드마크 중 검지 끝 — 몸 관절보다 훨씬 작음


def _make_hand_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def probe_hand_reliability(video_path: str, handedness: str = "Right") -> dict:
    """전신 촬영 거리에서 손가락 keypoint가 얼마나 믿을 만한지 진단한다.

    같은 영상의 오른손목(몸 관절, 이미 파이프라인에 있음) 검출 떨림과 나란히 비교해
    "손가락을 추가할 가치가 있는가"를 판단하는 근거로 쓴다 — papers.md G2/G3가
    지적한 근접 촬영 vs 전신 촬영 정확도 차이가 이 프로젝트 데이터에서도 나타나는지 확인.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    diag = (w ** 2 + h ** 2) ** 0.5

    pose_landmarker = _make_landmarker()
    hand_landmarker = _make_hand_landmarker()
    total = 0
    hand_positions: list[tuple[float, float]] = []
    wrist_positions: list[tuple[float, float]] = []
    try:
        frame_i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_i / fps) * 1000)
            total += 1

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            fingertip = None
            for hand, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
                if handed[0].category_name == handedness:
                    fingertip = hand[INDEX_FINGERTIP]
                    break
            if fingertip is not None:
                hand_positions.append((fingertip.x * w / diag, fingertip.y * h / diag))

            pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
            if pose_result.pose_landmarks:
                p = pose_result.pose_landmarks[0][LANDMARK_INDEX["오른손목"]]
                wrist_positions.append((p.x * w / diag, p.y * h / diag))

            frame_i += 1
    finally:
        cap.release()
        pose_landmarker.close()
        hand_landmarker.close()

    def jitter(positions: list[tuple[float, float]]) -> float:
        arr = np.array(positions)
        return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean()) if len(arr) > 1 else float("nan")

    return {
        "total_frames": total,
        "hand_detection_rate": len(hand_positions) / total if total else 0.0,
        "hand_frame_to_frame_jitter": jitter(hand_positions),
        "wrist_detection_rate": len(wrist_positions) / total if total else 0.0,
        "wrist_frame_to_frame_jitter": jitter(wrist_positions),
    }
