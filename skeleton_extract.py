"""쌤쌤 스타일 전이용 전신 3D 스켈레톤 추출 — MediaPipe PoseLandmarker world landmarks 기반.

pose_extract.py의 extract_joint_dynamics()는 8개 관절의 2D 속도/가속도/저크만 뽑는
식별 트랙 전용 함수라 스타일 전이엔 못 쓴다(samsam_dev_spec.md 0절). Motion Puzzle 등
스타일 전이 파이프라인은 33개 전체 랜드마크의 3D 위치 시퀀스가 필요해 별도 함수로 뽑는다.
관절 계층·회전은 여기서 만들지 않는다 — 그건 이후 SMPL/BVH 변환 단계(4-0) 몫이고, 여기는
원시 3D 랜드마크 좌표만 책임진다.

사용법:
  python skeleton_extract.py A_1.mp4 --out A_1_skeleton.npz
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
NUM_LANDMARKS = 33


def _make_landmarker() -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_skeleton_sequence(video_path: str) -> dict:
    """영상 → {"landmarks": (frame, 33, 3) 월드 좌표 배열, "fps": float}.

    world_landmarks는 힙 중심 원점의 미터 단위 3D 좌표(카메라 화면 좌표가 아님) —
    스타일 전이용 스켈레톤 시퀀스의 원재료. 검출 실패 프레임은 직전 프레임 값을
    유지한다(pose_extract.py와 동일한 정책) — 값이 통째로 비면 이후 BVH 변환에서
    그 프레임만 깨지는 게 아니라 시퀀스 길이 자체가 흔들린다.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = _make_landmarker()
    frames: list[np.ndarray] = []
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
            world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            if world is not None:
                frames.append(np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64))
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((NUM_LANDMARKS, 3)))
            frame_i += 1
    finally:
        cap.release()
        landmarker.close()

    if frame_i < 10:
        raise ValueError(f"동작 추출 실패 (프레임 부족): {video_path}")

    return {"landmarks": np.stack(frames), "fps": fps}


def save_skeleton_npz(video_path: str, out_path: str) -> None:
    data = extract_skeleton_sequence(video_path)
    np.savez(out_path, landmarks=data["landmarks"], fps=data["fps"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or (Path(args.video).stem + "_skeleton.npz")
    save_skeleton_npz(args.video, out)
    print(f"저장 완료: {out}")
