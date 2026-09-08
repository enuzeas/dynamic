"""ICC(3,1) 검정력(power) 분석 — 표본 크기가 추정치를 얼마나 흔드는지 몬테카를로로 확인.

"진짜" ICC가 이렇다고 가정(우리가 실측한 M02/M04/M06 값을 앵커로 씀)하고, 그 가정 아래서
다양한 (n=사람 수, k=반복 수) 표본을 수천 번 뽑아 icc_3_1 추정치가 얼마나 흔들리는지 본다.
evaluate_icc.py의 icc_3_1을 그대로 재사용한다(공식을 새로 안 만듦).

생성 모델(ICC(3,1)이 가정하는 이원배치 모형의 단순화): X_ij = P_i + E_ij
  P_i ~ N(0, sqrt(true_icc))   — 사람 i의 고유 성분(분산 = true_icc)
  E_ij ~ N(0, sqrt(1-true_icc)) — 반복마다의 잡음(분산 = 1-true_icc)
전체 분산을 1로 정규화했으므로 Var(P)/[Var(P)+Var(E)] = true_icc가 정확히 성립한다.

이 스크립트는 "우리 데이터의 진짜 ICC가 얼마다"를 말해주지 않는다 — 이미 실측한 값을
가정으로 넣고 "그 가정이 맞다면 표본을 이만큼 모았을 때 추정치가 얼마나 믿을만한가"만
알려준다(검정력 분석의 원래 용도).

사용법: python icc_power_analysis.py
"""
from __future__ import annotations

import numpy as np

from evaluate_icc import icc_3_1

# 실측 앵커 — track1_icc_pilot.md 결과표에서 그대로 가져옴
TRUE_ICC_CASES = [
    ("M06 조깅 실측치", 0.870),
    ("M04 앉았다 일어서기 실측치", 0.774),
    ("M02 손 뻗기 실측치", 0.491),
]

# (설명, n=사람 수, k=반복 수)
SCENARIOS = [
    ("현재 M06 표본 (P005 제외)", 6, 2),
    ("현재 M04 표본", 7, 3),
    ("P008~015 정리 후, S2 전 (전원 S1만)", 15, 3),
    ("S2 재촬영 대상 5명만 (S1+S2 합쳐 반복↑)", 5, 5),
    ("scope.md 목표 (8~12명, 인당 5회 중간값)", 10, 5),
    ("S2를 15명 전체로 확대", 15, 5),
]

N_SIMS = 3000


def simulate_icc_estimates(true_icc: float, n: int, k: int, n_sims: int = N_SIMS, seed: int = 0) -> np.ndarray:
    """true_icc를 정답으로 두고 (n,k) 표본을 n_sims번 뽑아 icc_3_1 추정치 배열을 반환."""
    rng = np.random.default_rng(seed)
    sigma_p, sigma_e = np.sqrt(true_icc), np.sqrt(1 - true_icc)
    estimates = np.empty(n_sims)
    for s in range(n_sims):
        person_effect = rng.normal(0, sigma_p, size=(n, 1))
        noise = rng.normal(0, sigma_e, size=(n, k))
        estimates[s] = icc_3_1(person_effect + noise)
    return estimates


if __name__ == "__main__":
    for case_label, true_icc in TRUE_ICC_CASES:
        print(f"\n{'='*70}\n가정: 진짜 ICC = {true_icc:.3f} ({case_label})\n{'='*70}")
        print(f"{'시나리오':40s} {'n':>3s} {'k':>3s} {'평균 추정치':>10s} {'95% 범위':>18s} {'P(추정<0.5)':>12s}")
        for label, n, k in SCENARIOS:
            est = simulate_icc_estimates(true_icc, n, k)
            lo, hi = np.percentile(est, [2.5, 97.5])
            p_below_half = float(np.mean(est < 0.5))
            print(f"{label:40s} {n:3d} {k:3d} {est.mean():10.3f} [{lo:6.3f}, {hi:6.3f}]  {p_below_half:11.1%}")
