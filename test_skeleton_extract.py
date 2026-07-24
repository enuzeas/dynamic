"""extract_skeleton_sequence 자체 검증 — 구조가 정상인지만 확인
(형태·값 범위만 본다, 실제 스타일 전이 품질 판단은 이후 Motion Puzzle 스파이크 몫).

실행: python test_skeleton_extract.py
"""
from skeleton_extract import NUM_LANDMARKS, extract_skeleton_sequence


def main() -> None:
    data = extract_skeleton_sequence("A_1.mp4")
    landmarks, fps = data["landmarks"], data["fps"]
    print(f"프레임 수: {landmarks.shape[0]}, fps: {fps}")

    assert landmarks.ndim == 3, f"(frame, joint, xyz) 3차원이어야 한다: {landmarks.shape}"
    assert landmarks.shape[1] == NUM_LANDMARKS, f"랜드마크 33개여야 한다: {landmarks.shape[1]}"
    assert landmarks.shape[2] == 3, f"xyz 3축이어야 한다: {landmarks.shape[2]}"
    assert landmarks.shape[0] > 10, "프레임이 너무 적다"
    assert fps > 0, f"fps는 양수여야 한다: {fps}"
    # world landmark는 힙 중심 미터 단위라 사람 한 명 크기(수 미터) 범위를 벗어나면 이상치다.
    assert abs(landmarks).max() < 10.0, f"좌표 범위가 비정상적으로 크다: {abs(landmarks).max()}"

    print("자체 검증 통과")


if __name__ == "__main__":
    main()
