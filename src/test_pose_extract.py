"""probe_hand_reliability 자체 검증 — 구조가 정상인지만 확인 (실제 판단은
doc/next.md #4에 기록된 실측 수치로 함, 여기선 크래시·값 범위만 본다).

실행 (repo 루트에서): python src/test_pose_extract.py
"""
from pose_extract import probe_hand_reliability


def main() -> None:
    result = probe_hand_reliability("sample/original/A_1.mp4")
    print(result)

    assert result["total_frames"] > 0
    for key in ("hand_detection_rate", "wrist_detection_rate"):
        assert 0.0 <= result[key] <= 1.0, f"{key}는 0~1 사이여야 한다: {result[key]}"
    for key in ("hand_frame_to_frame_jitter", "wrist_frame_to_frame_jitter"):
        assert result[key] >= 0.0, f"{key}는 음수일 수 없다: {result[key]}"

    print("\n자체 검증 통과")


if __name__ == "__main__":
    main()
