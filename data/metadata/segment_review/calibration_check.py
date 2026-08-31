"""M00(캘리브레이션) 테이크 자동 스크리닝 — calibration_ok 채우기 전 1차 확인용.

체크리스트(쌤쌤_촬영_체크리스트_v2.md §8)의 "T포즈 5초·중립 정지 10초를 제대로
유지했는가"를 사람이 보지 않고 근사한다: 관절 검출률(카메라 밖으로 나갔거나
가려짐)과 프레임 이탈 여부만 본다. 최종 판단은 사람이 한다 — 여기선 후보만 거른다.

실행 (repo 루트에서): .venv/bin/python data/metadata/segment_review/calibration_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_lite.task"
RAW_DIR = REPO_ROOT / "data" / "raw" / "_nosound"

JOINTS = (16, 14, 12, 15, 13, 11, 24, 23)  # pose_extract.LANDMARK_INDEX 값과 동일


def check_clip(video_path: Path) -> dict:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    total = 0
    detected = 0
    in_frame = 0
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
            total += 1
            frame_i += 1
            if not result.pose_landmarks:
                continue
            lm = result.pose_landmarks[0]
            pts = np.array([(lm[j].x, lm[j].y) for j in JOINTS])
            detected += 1
            if np.all((pts >= -0.02) & (pts <= 1.02)):  # 살짝 여유
                in_frame += 1
    finally:
        cap.release()
        landmarker.close()

    return {
        "file": video_path.name,
        "total_frames": total,
        "detection_rate": detected / total if total else 0.0,
        "in_frame_rate": in_frame / total if total else 0.0,
    }


def main() -> None:
    clips = sorted(RAW_DIR.glob("*_M00_*.mov"))
    if not clips:
        print(f"M00 클립을 못 찾음: {RAW_DIR}")
        sys.exit(1)

    print(f"{'file':40s} {'frames':>7s} {'detect%':>8s} {'in_frame%':>10s}  판정")
    for clip in clips:
        r = check_clip(clip)
        ok = r["detection_rate"] >= 0.95 and r["in_frame_rate"] >= 0.95
        verdict = "Y (자동 통과)" if ok else "N → 사람이 재확인 필요"
        print(
            f"{r['file']:40s} {r['total_frames']:7d} "
            f"{r['detection_rate']*100:7.1f}% {r['in_frame_rate']*100:9.1f}%  {verdict}"
        )


if __name__ == "__main__":
    main()
