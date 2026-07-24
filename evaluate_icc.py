"""세션 6-2 항상성 정량 평가 — ICC(3,1) 구현.

papers.md F1(Koo & Li 2016 가이드라인)·F2(Shrout & Fleiss 1979 원전) 근거.
"같은 사람 반복 촬영 간 항상성"을 보는 게 목적이라 rater(반복 촬영 차수)의
체계적 차이는 무시하는 "consistency" 모델 — ICC(2,1)의 absolute agreement가
아니라 ICC(3,1)을 쓴다(analyze.py의 "항상성 검증" 관점과 동일한 의도).

사용법:
  python evaluate_icc.py --npz features.npz   # features: (n_subjects, k_repeats)
"""
from __future__ import annotations

import numpy as np


def icc_3_1(data: np.ndarray) -> float:
    """ICC(3,1), 이원배치 혼합모형·consistency·단일측정.

    data: (n_subjects, k_repeats) — 사람마다 반복 촬영 횟수가 같아야 한다
    (unbalanced design은 범위 밖 — ponytail: 균형설계 가정, 반복 횟수가 사람마다
    다르면 이 함수로 못 쓰고 REML 기반 확장이 필요하다).
    """
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError(f"최소 2명 × 2반복 필요: n={n}, k={k}")

    grand_mean = data.mean()
    row_means = data.mean(axis=1)  # 사람별 평균
    col_means = data.mean(axis=0)  # 반복 차수별 평균

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if denom == 0:
        return 1.0 if ms_rows == 0 else 0.0
    return (ms_rows - ms_error) / denom


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help='"features" 키에 (n_subjects, k_repeats) 배열 저장된 npz')
    args = parser.parse_args()

    data = np.load(args.npz)["features"]
    icc = icc_3_1(data)
    print(f"ICC(3,1) = {icc:.3f}  (n={data.shape[0]}명, k={data.shape[1]}반복)")
