"""resample_fraction / between_subject_r 자체 검증.

핵심 회귀 방지 대상: DTW 기반 정렬은 타인 간 상관까지 0.85+로 밀어올려서
폐기했다(실측 확인). percentage-of-duration 방식은 최소한 그 정도로
"뭐든 비슷하게" 만들지는 않아야 한다 — 완전히 다른 두 파형을 억지로
맞추면 안 된다는 것만 확인한다. (진짜 within > between 분리는 실제
샘플로만 확인 가능 — 지금 데이터에선 분리가 약하다는 게 알려진 상태.)

실행 (repo 루트에서): python src/test_homeostasis_image.py
"""
import numpy as np

from homeostasis_image import between_subject_r, pearsonr, resample_fraction

FPS = 30.0


def main() -> None:
    # 길이가 다른 두 반복이라도 리샘플 후 같은 길이가 되는지
    a = np.sin(np.linspace(0, 2 * np.pi, 60))
    b = np.sin(np.linspace(0, 2 * np.pi, 90))
    ra, rb = resample_fraction(a), resample_fraction(b)
    assert len(ra) == len(rb) == 50, "리샘플 후 길이가 고정 길이(n)와 달라야 하면 버그"

    # 진짜 같은 파형(템포만 다름)은 리샘플 후 거의 완전히 같아야 함
    same_wave_r = pearsonr(ra, rb)
    print(f"같은 파형, 템포만 다름: r={same_wave_r:.3f}")
    assert same_wave_r > 0.95, "시간 비율 정규화만으로도 템포 차이는 흡수돼야 한다"

    # 서로 완전히 다른 두 사람(파형 자체가 다름) 흉내 — DTW처럼 인위적으로
    # 비슷해지면 안 된다
    t = np.linspace(0, 1, 50)
    fake_person_1 = {"p1": [np.sin(2 * np.pi * t) for _ in range(3)]}
    fake_person_2 = {"p2": [np.sin(2 * np.pi * t + np.pi) + 0.5 * np.cos(6 * np.pi * t) for _ in range(3)]}
    fake_people = {**fake_person_1, **fake_person_2}
    r_between = between_subject_r(fake_people)
    print(f"파형이 뚜렷이 다른 두 사람의 타인 간 상관: r={r_between:.3f}")
    assert r_between < 0.5, "정렬 방법이 서로 다른 파형까지 억지로 비슷하게 만들면 안 된다"

    print("\n자체 검증 통과")


if __name__ == "__main__":
    main()
