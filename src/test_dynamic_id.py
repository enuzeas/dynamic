"""Dynamic ID 파이프라인 자체 검증 — 저장소의 실제 샘플 영상(A_*, B_*)으로 등록·검증·식별을 끝까지 돌린다.

실행 (repo 루트에서): python src/test_dynamic_id.py
"""
from dynamic_id import calibrate_threshold, enroll, identify, verify


def main() -> None:
    id_a = enroll("A", ["sample/original/A_1.mp4", "sample/original/A_2.mp4"])
    id_b = enroll("B", ["sample/original/B_1.mp4", "sample/original/B_2.mp4"])
    registry = [id_a, id_b]

    assert set(id_a.joints) == set(id_b.joints)
    assert len(id_a.templates["오른손목"]) == 2, "등록 반복 2회가 템플릿에 그대로 들어가야 한다"

    threshold = calibrate_threshold(registry)
    print(f"보정된 임계값: {threshold:.2f}")

    accepted, distance = verify("sample/original/A_3.mp4", id_a, threshold)
    print(f"A_3 vs A 등록 — 거리 {distance:.2f} (임계값 {threshold:.2f}) → {'accept' if accepted else 'reject'}")

    label, distance = identify("sample/original/A_3.mp4", registry)
    print(f"A_3 identify 결과: {label} (거리 {distance:.2f})")
    assert label == "A", f"A_3은 A로 식별돼야 하는데 {label}로 나왔다"

    label_b, _ = identify("sample/original/B_3.mp4", registry)
    print(f"B_3 identify 결과: {label_b}")
    assert label_b == "B", f"B_3은 B로 식별돼야 하는데 {label_b}로 나왔다"

    print("\n모든 자체 검증 통과")


if __name__ == "__main__":
    main()
