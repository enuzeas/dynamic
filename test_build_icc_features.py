"""build_icc_features 자체 검증 — 실제 영상·mediapipe 없이 배열 조립 로직만 확인.

실행 (build_icc_features가 mediapipe/cv2를 임포트하므로 해당 venv 필요):
  .venv/bin/python test_build_icc_features.py
"""
import numpy as np

from build_icc_features import _trim_window, build_features
from demo import trim_motion


def main() -> None:
    subjects = {
        "A": ["a1.mp4", "a2.mp4"],
        "B": ["b1.mp4", "b2.mp4"],
    }
    fake_values = {"a1.mp4": 1.0, "a2.mp4": 2.0, "b1.mp4": 5.0, "b2.mp4": 6.0}

    names, features = build_features(subjects, extract_fn=lambda path: fake_values[path])

    assert names == ["A", "B"], names
    assert features.shape == (2, 2), features.shape
    assert np.array_equal(features, np.array([[1.0, 2.0], [5.0, 6.0]])), features

    # _trim_window는 trim_motion과 같은 구간을 슬라이스로만 반환해야 한다(다른 시계열에
    # 재사용하려고 분리한 것 — speed 자신에 적용했을 때 원본과 결과가 같아야 신뢰할 수 있다).
    speed = np.array([0.01, 0.02, 0.5, 0.9, 0.4, 0.05, 0.01, 0.02])
    assert np.array_equal(speed[_trim_window(speed)], trim_motion(speed))

    print("자체 검증 통과")


if __name__ == "__main__":
    main()
