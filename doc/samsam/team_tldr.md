# 쌤쌤 TL;DR — 팀 공유용 (2026-08-06 기준)

> 자세한 내용 다 안 읽어도 되게 요약. 궁금한 부분만 아래 링크 따라가면 됨.

---

## 1. 가장 중요한 것 — 교수 미팅 결과 (2026-08-05)

Motion Puzzle(우리가 스타일 전이에 쓰는 핵심 모델)을 만든 **이성희 교수님(KAIST)**께 직접 자문받았다.

- **"스타일"과 "습관"은 다른 문제라는 답을 받음.** 스타일 = 걷기 하나에서 나타나는 느낌, 습관 = 걷든 뛰든 손짓하든 변하지 않는 그 사람다움. 우리가 원래 하려던 건 후자(습관)인데, 지금 설계(Motion Puzzle 위에 정제 헤드 얹기)가 그 문제에 맞는 방법인지는 **재검토가 필요**해졌다. 아직 답이 아니라 "다시 생각해봐야 함" 단계.
- **검증 방법은 통과.** "같은 사람인지 구별하기" + "동작 하나 빼고도 그 사람을 알아맞히기(LOMO)" 방식은 "개인 고유성 검증엔 적절하다"고 확인받음 — 이건 그대로 간다.
- **표정 트랙은 보류.** 얼굴은 몸짓과 다른 영역이고 이 교수님 랩 소관도 아니라서, 하나에 집중하라는 조언을 받아 몸짓에만 집중하기로 함.
- **대학원생 한 분이 배정돼 앞으로 같이 개발한다.** 정기 미팅 예정.

## 2. 그래서 다음에 정해야 할 것

대학원생과의 첫 정기 미팅에서 방향을 정해야 함:
1. 원래 계획대로 "습관 추출"을 밀고 갈지
2. 스코프를 줄여서 "한 동작 안에서의 개인차"만 볼지
3. 남은 기간(~11월)이 짧으니 교수님이 "더 수월하다"고 한 "개인 식별 분류기" 쪽으로 무게중심을 옮길지
4. (2026-08-06 신규) 아래 3절에서 찾은 "가벼운 실무 버전"으로 갈지

> ⚠️ **이 목록은 결론이 아니라 재료다.** 특히 3·4번은 "짧은 기간에 확실히 됨"이라 매력적으로 보이지만, 그만큼 원래 하려던 것 — "그 사람 습관이 느껴지는 살아있는 캐릭터" — 과는 거리가 있다. **쉬워 보인다고 먼저 정해버리지 말 것** — 대학원생과 함께 트레이드오프를 놓고 판단해야 하는 자리다.

## 3. (2026-08-06) "습관 추출", 실제로 되는 방법이 있는지 찾아봤다

헷갈리기 쉬운 세 개념 정리:

| | 스타일 추출 | 한 동작 안 개인차 | 습관 추출 |
|---|---|---|---|
| 뭘 고정하는가 | 클립 1개 | 동작 종류(예: 걷기만) | 안 고정 — 동작을 넘나듦 |
| 지금 상태 | 이미 있음(Motion Puzzle 기본 기능) | 새 학습 없이 바로 시도 가능 | 학계도 못 검증한 영역, 리스크 큼 |

- **스타일만 쓰면 Motion Puzzle을 그대로 쓰는 것과 같음** — 우리 기여는 "한 동작 안 개인차" 검증(ICC+재식별)부터 생김. Motion Puzzle 원저자들도 안 해본 검증(README에 Todo로 남아있음).
- **습관 신호는 문헌상 "얇다"** — 여러 동작을 넘나들며 유지된다고 확인된 건 복잡한 패턴이 아니라 vigor(움직임 크기)·변동성 같은 숫자 1~2개 정도.
- **그래서 나온 실무 대안**: 이 숫자를 딥러닝으로 학습시키지 않고, 완성된 캐릭터 위에 바로 얹는 "다이얼"로 쓰는 것(예: vigor 높으면 더 크고 빠르게 움직임). 설명하기 쉽고 지금 만들 수 있지만, **"습관 재현"이라는 원래 그림의 축소판**이라는 건 알고 골라야 함.
- 자세히: [papers.md](https://github.com/enuzeas/dynamic/blob/main/doc/bio-ip-archive/papers.md) A절(A4·A9·A11) / [motion_ip_pipeline.md](https://github.com/enuzeas/dynamic/blob/main/doc/samsam/motion_ip_pipeline.md) 7장 0번(d).

## 4. 촬영 준비 — 실무

- **체크리스트 v2** 나옴 — 캘리브레이션(T포즈 등록) 추가, 손 뻗기 동작 추가돼서 총 8동작 세트로 확장. 1인당 약 18분.
- **카메라: G9II(메인) + GoPro5(보조)** 예정. 주의할 점:
  - GoPro는 기본 렌즈가 어안이라 왜곡 심함 → **1080p + Linear 모드**로 찍기
  - 두 카메라 다 **손떨림보정 OFF** (프레임 좌표계 흔들리면 정규화 깨짐)
  - 가능하면 **120fps** — 저크(가속도 변화) 신호가 프레임레이트에 민감해서 여유 있으면 60fps보다 안전
  - GoPro 쪽 영상이 노이즈 더 낄 수 있어서, 촬영 첫날 G9II·GoPro 신뢰도 비교 한 번 해볼 것

## 5. 더 자세히 보려면

| 궁금한 것 | 어디를 볼지 |
|---|---|
| 미팅 실제 대화 내용 | [External_doc/20260805_이성희교수_회의록.md](https://github.com/enuzeas/dynamic/blob/main/External_doc/20260805_%EC%9D%B4%EC%84%B1%ED%9D%AC%EA%B5%90%EC%88%98_%ED%9A%8C%EC%9D%98%EB%A1%9D.md) |
| 미팅 결과가 기술 설계에 어떻게 반영됐는지 | [doc/samsam/motion_ip_pipeline.md](https://github.com/enuzeas/dynamic/blob/main/doc/samsam/motion_ip_pipeline.md) 1-1장·7장 0번 |
| "습관 추출" 관련 논문 목록 | [doc/bio-ip-archive/papers.md](https://github.com/enuzeas/dynamic/blob/main/doc/bio-ip-archive/papers.md) A절 |
| 항목별 진행 상태 (뭐가 됐고 뭐가 안 됐는지) | [doc/samsam/professor_review_status.md](https://github.com/enuzeas/dynamic/blob/main/doc/samsam/professor_review_status.md) |
| 촬영 현장에서 뭘 준비해야 하는지 | [External_doc/쌤쌤_촬영_체크리스트_v2.md](https://github.com/enuzeas/dynamic/blob/main/External_doc/%EC%8C%A4%EC%8C%A4_%EC%B4%AC%EC%98%81_%EC%B2%B4%ED%81%AC%EB%A6%AC%EC%8A%A4%ED%8A%B8_v2.md) |
| 전체 프로젝트 개요 | [README.md](https://github.com/enuzeas/dynamic/blob/main/README.md) |
