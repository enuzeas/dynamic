# 기술/코드 TODO

`workthrough.md`(코드), `doc/plan.md`(실행 계획), `doc/direction_brief.md`(현재 상태)를 대조해서 나온, 아직 안 끝난 것 위주. 우선순위 순.

## 1. 모델 파일 없이는 실행 자체가 안 됨 (즉시 막힘)
`pose_extract.py`가 `models/pose_landmarker_lite.task`를 읽는데(pose_extract.py:17), `models/` 디렉터리가 로컬에 없고 `.gitignore`의 `*.task`에 걸려 저장소에도 없다. 루트의 `pose_landmarker.task`는 이름·경로가 다른 별개 파일이라 대신 쓸 수 없다.
- MediaPipe Pose Landmarker lite 모델을 받아 `models/pose_landmarker_lite.task`에 배치
- README나 setup 스크립트에 "모델 파일 받아야 함" 한 줄 남기기 (지금 아무 문서에도 없음)

## 2. ICC와 rank-1의 정렬 기준 통일 (`doc/plan.md` 3절, 아직 미반영)
`analyze.py`는 DTW로 정렬해서 rank-1/EER을 내고, `homeostasis_image.py`는 `trim_motion` 후 최단 길이로 자르는 정렬만 쓴다. plan.md가 지목한 "고유성은 잘 잡히는데 항상성은 안 잡히는" 역설의 원인 중 하나로 의심되는 지점인데, MediaPipe+평활화 전환 후에도 아직 코드가 안 바뀌었다.
- `homeostasis_image.py`의 정렬 로직을 DTW 기반으로 바꾸고 ICC 재측정
- 전환 전 수치(ICC 0.05~0.07)가 MediaPipe 전환 후에도 그대로인지부터 재실행해서 확인

## 3. 관절 수 확장 (3개 → scope.md MVP 스펙 6~8개)
`pose_extract.LANDMARK_INDEX`는 오른손목/팔꿈치/어깨 세 관절뿐. plan.md는 scope.md 기준 6~8개 관절(다관절 결합)을 MVP 스펙으로 명시한다.
- 반대편 팔·골반·다리 등 추가 관절을 `LANDMARK_INDEX`에 넣고 `dynamic_id.py`의 관절 평균 거리 계산이 그대로 확장되는지 확인

## 4. 표본 확대 + Phase 0 데이터 거버넌스 (plan.md Phase 0~1)
직접 촬영 8~12명·인당 5회, 유튜브는 라이선스 확인된 채널만 우선 파일럿 3~5개 — 아직 저장소엔 3명(A/B/C) 수준 샘플뿐.
- `sample/` 확대는 촬영·동의서 정리가 먼저(성진구 담당 영역, 코드 작업 아님)
- 유튜브 소스는 라이선스 기준 정리 전까지 `sample_dance/` 확장하지 않기

## 5. 잡다한 정리
- `requirements.txt`/`pyproject.toml` 없음 — `mediapipe`, `scipy`, `opencv-python`, `numpy`, `matplotlib` 버전 고정 필요 (재현성)
- `homeostasis_image.py`는 CLI 인자가 없고 `PEOPLE`이 하드코딩 — `dynamic_id.py`처럼 인자로 뺄지 결정 (2번 작업과 같이 하면 자연스러움)
