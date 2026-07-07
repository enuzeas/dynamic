"""Eulerian Video Magnification — 육안으로 안 보이는 미세 움직임을 증폭해서 보여준다.

Wu et al. 2012의 방식을 단순화한 버전: 그레이스케일 다운샘플 → 시간축 버터워스
밴드패스 필터 → 증폭 → 원본에 더해서 재구성. 색상 채널별 가중치, 다중 스페이셜
밴드(라플라시안 피라미드 레벨별 증폭)는 생략했다 — 이 프로젝트가 보고 싶은 건
맥박이 아니라 "동작 사이 미세 흔들림이 실제로 있는가"라 단일 스케일로 충분하다.
필요해지면 raw pixel-level 색상 증폭으로 확장.

사용법:
  python eulerian_magnify.py --video A_1.mp4 --alpha 20 --low 0.5 --high 3.0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def load_gray_frames(video_path: str, max_height: int = 320) -> tuple[np.ndarray, float]:
    """영상 → (T, H, W) 그레이스케일 배열. 계산량을 줄이려고 max_height로 다운샘플."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_height / h)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float64))
    cap.release()

    if len(frames) < 8:
        raise ValueError(f"프레임 부족 (밴드패스 필터에는 최소 8프레임 필요): {video_path}")
    return np.stack(frames), fps


def magnify_signal(frames: np.ndarray, fps: float, low: float, high: float,
                    alpha: float, order: int = 4) -> np.ndarray:
    """(T, H, W) 시계열에 시간축 버터워스 밴드패스를 걸고 alpha배 증폭한 성분만 반환."""
    t, h, w = frames.shape
    nyquist = fps / 2
    b, a = butter(order, [low / nyquist, min(high / nyquist, 0.99)], btype="band")
    flat = frames.reshape(t, h * w)
    band = filtfilt(b, a, flat, axis=0)
    return (band * alpha).reshape(t, h, w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--alpha", type=float, default=20.0, help="증폭 배수")
    parser.add_argument("--low", type=float, default=0.5, help="밴드패스 하한 (Hz)")
    parser.add_argument("--high", type=float, default=3.0, help="밴드패스 상한 (Hz)")
    parser.add_argument("--out-video", default="eulerian_output.mp4")
    parser.add_argument("--out-plot", default="eulerian_output.png")
    args = parser.parse_args()

    print(f"프레임 로딩: {args.video}")
    frames, fps = load_gray_frames(args.video)
    t, h, w = frames.shape
    print(f"  {t}프레임, {w}x{h}, {fps:.1f}fps ({t/fps:.1f}초)")

    print(f"밴드패스 필터링 + {args.alpha}배 증폭 ({args.low}~{args.high}Hz)...")
    amplified = magnify_signal(frames, fps, args.low, args.high, args.alpha)
    magnified_frames = np.clip(frames + amplified, 0, 255).astype(np.uint8)

    # ── 증폭 영상 저장 ──────────────────────────────────
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    for f in magnified_frames:
        writer.write(f)
    writer.release()
    print(f"저장 완료: {args.out_video}")

    # ── 가장 변동이 큰 지점 하나를 골라 raw vs 증폭 비교 ──
    variance_map = amplified.var(axis=0)
    py, px = np.unravel_index(np.argmax(variance_map), variance_map.shape)
    raw_trace = frames[:, py, px] - frames[:, py, px].mean()
    amp_trace = amplified[:, py, px]
    time = np.arange(t) / fps

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#080A10")
    ax.set_facecolor("#0F1219")
    ax.plot(time, raw_trace, color="#6A7A99", linewidth=1.2, label="원본 (밝기 변화, 평균 제거)")
    ax.plot(time, amp_trace, color="#D4451E", linewidth=1.6, label=f"증폭 후 ({args.alpha}배, {args.low}-{args.high}Hz)")
    ax.set_title(f"Eulerian 미세 움직임 증폭 — 픽셀 ({px},{py})", color="#D4DEEE", pad=10)
    ax.set_xlabel("시간 (초)", color="#6A7A99", fontsize=9)
    ax.set_ylabel("밝기 변화량", color="#6A7A99", fontsize=9)
    ax.tick_params(colors="#6A7A99", labelsize=8)
    ax.spines[:].set_color("#1E2840")
    ax.legend(fontsize=9, labelcolor="#D4DEEE", facecolor="#141720", edgecolor="#1E2840")
    plt.savefig(args.out_plot, dpi=150, bbox_inches="tight", facecolor="#080A10")
    print(f"저장 완료: {args.out_plot}")


if __name__ == "__main__":
    main()
