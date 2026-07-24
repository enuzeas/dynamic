"""icc_3_1 자체 검증 — 알려진 극단 케이스로 공식이 맞는지 확인.

실행: python test_evaluate_icc.py
"""
import numpy as np

from evaluate_icc import icc_3_1


def main() -> None:
    # 완벽히 일관된 데이터(반복마다 사람별 값이 동일) → ICC = 1
    perfect = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
    icc_perfect = icc_3_1(perfect)
    print(f"완벽 일관 케이스: {icc_perfect:.3f}")
    assert abs(icc_perfect - 1.0) < 1e-9, f"완벽 일관 데이터는 ICC=1이어야 한다: {icc_perfect}"

    # 사람 간 진짜 차이(1.5 vs 4.5)보다 반복 간 흔들림이 그만큼 커서 신호와 잡음이 같은 크기
    # → ICC = 0 (손으로 계산해 검증한 값, samsam_dev_spec.md 참고 없이 직접 유도)
    noisy = np.array([[1.0, 2.0], [1.0, 8.0], [1.0, 2.0], [1.0, 8.0]])
    icc_zero = icc_3_1(noisy)
    print(f"신호=잡음 케이스: {icc_zero:.3f}")
    assert abs(icc_zero) < 1e-9, f"이 구성에서는 ICC가 정확히 0이어야 한다: {icc_zero}"

    print("자체 검증 통과")


if __name__ == "__main__":
    main()
