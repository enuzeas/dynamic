"""icc_power_analysis 자체 검증 — 표본이 커지면 추정치가 진짜 값에 가까워지고
흔들림(표준편차)이 줄어드는지 확인한다(몬테카를로 생성 모델이 실제로 true_icc를
그 이름값대로 만들어내는지에 대한 sanity check).

실행: python test_icc_power_analysis.py
"""
import numpy as np

from icc_power_analysis import simulate_icc_estimates


def main() -> None:
    true_icc = 0.7

    small = simulate_icc_estimates(true_icc, n=5, k=2, n_sims=2000, seed=1)
    large = simulate_icc_estimates(true_icc, n=200, k=10, n_sims=2000, seed=1)

    # 표본이 크면 평균 추정치가 진짜 값에 훨씬 가까워야 한다.
    assert abs(large.mean() - true_icc) < 0.02, large.mean()
    assert abs(small.mean() - true_icc) < 0.15, small.mean()

    # 표본이 크면 추정치가 덜 흔들려야 한다(표준편차 감소).
    assert large.std() < small.std() / 3, (small.std(), large.std())

    print(f"n=5,k=2   평균={small.mean():.3f} 표준편차={small.std():.3f}")
    print(f"n=200,k=10 평균={large.mean():.3f} 표준편차={large.std():.3f}")
    print("자체 검증 통과")


if __name__ == "__main__":
    main()
