"""magnify_signal 자체 검증 — 합성 신호로 밴드 안/밖 주파수가 다르게 처리되는지 확인.

실행 (repo 루트에서): python src/test_eulerian_magnify.py
"""
import numpy as np

from eulerian_magnify import magnify_signal

FPS = 30.0


def main() -> None:
    t = np.arange(300) / FPS  # 10초
    in_band = np.sin(2 * np.pi * 1.0 * t)      # 1.0Hz — 밴드(0.5~3Hz) 안
    out_band = np.sin(2 * np.pi * 6.0 * t)     # 6.0Hz — 밴드 밖

    frames = np.zeros((300, 2, 1))
    frames[:, 0, 0] = in_band
    frames[:, 1, 0] = out_band

    alpha = 10.0
    amplified = magnify_signal(frames, FPS, low=0.5, high=3.0, alpha=alpha)

    in_band_gain = amplified[:, 0, 0].std() / in_band.std()
    out_band_gain = amplified[:, 1, 0].std() / out_band.std()

    print(f"밴드 안(1.0Hz) 증폭 배수: {in_band_gain:.2f} (목표 alpha={alpha})")
    print(f"밴드 밖(6.0Hz) 증폭 배수: {out_band_gain:.2f} (억제되어야 함)")

    assert in_band_gain > alpha * 0.7, "밴드 안 주파수가 충분히 증폭되지 않았다"
    assert out_band_gain < alpha * 0.3, "밴드 밖 주파수가 억제되지 않았다"
    print("\n자체 검증 통과")


if __name__ == "__main__":
    main()
