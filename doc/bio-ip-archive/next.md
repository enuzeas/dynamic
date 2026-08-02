# 기술/코드 TODO

`workthrough.md`(코드), `doc/plan.md`(실행 계획), `doc/direction_brief.md`(현재 상태)를 대조해서 나온, 아직 안 끝난 것 위주. 우선순위 순.

## 1. 모델 파일 (로컬은 해결, 새 클론은 아직 막힘)
루트의 `pose_landmarker.task`(5.7MB)가 실제로 MediaPipe Pose Landmarker **lite** 모델이었다 — `src/models/pose_landmarker_lite.task`로 복사하니 `extract_joint_dynamics`가 바로 동작함(A_1.mp4 186프레임 전부 검출 확인). 다운로드 불필요, 파일 위치만 문제였음.
- `pose_landmarker.task`, `src/models/pose_landmarker_lite.task` 둘 다 `.gitignore`의 `*.task`에 걸려 git에 없다 — 새로 클론하면 이 파일 자체가 없다. 저장소 밖(드라이브 등)에서 모델 파일을 공유하는 경로를 정하고, README/setup에 "받아서 `src/models/pose_landmarker_lite.task`로 복사" 한 줄 남기기 (README에 반영 완료)

## 1.5. `trim_motion` threshold 스케일 버그 — 발견 및 수정 완료
실제로 돌려보니 `trim_motion`의 기본 threshold(0.08)가 옛 옵티컬플로우 속도 스케일 기준이었다. MediaPipe 정규화 좌표 기반 속도는 영상마다 최대치가 0.08~0.11 수준이라, 절대 threshold 0.08은 분포 최상단이라 대부분의 프레임이 잘려나갔다(A_1.mp4: 186프레임 → 7프레임). 이 때문에 rank-1 66.7%→분석 시 16.7%, EER 47.2%(거의 랜덤)로 나왔던 것.
- `demo.py`의 `trim_motion` threshold를 절대값 → **최고 속도 대비 비율**로 변경(기본 0.15), `homeostasis_image.py`의 `THRESHOLD`도 동일하게 수정 — 완료
- 수정 후 재실행(A/B 6샘플): **rank-1 66.7%, EER 33.3%** (버그 상태 16.7%/47.2%에서 회복). 여전히 표본이 너무 작아 확정 수치는 아님

## 2. ICC 정렬 방법 — DTW는 폐기, percentage-of-duration으로 교체, 결론: 신호 자체가 약함
threshold 버그를 고친 뒤에도 `homeostasis_image.py`의 ICC는 **0.0000**(3명 전부)이었다. plan.md 3절 제안대로 DTW 기반 정렬을 시도했더니 ICC가 0.86~0.89로 튀었지만, **타인 간(다른 사람) 데이터를 같은 방식으로 정렬해봤더니 그 상관도 0.83~0.95로 똑같이 높았다** — DTW가 "같은 사람이라 비슷한" 게 아니라 "정렬이 뭐든 비슷하게 만드는" 것이었다. 이 프로젝트의 전제(타이밍 차이=개인 서명)와 정면으로 충돌하는 방법이라 폐기.
- percentage-of-duration 리샘플링(시간을 0~100% 진행률로만 맞춤, 국소 타이밍은 보존)으로 교체 — 완료. 타인 간 대조군(`between_subject_r`)도 스크립트에 상시 출력하도록 추가해, 앞으로 어떤 정렬법을 쓰든 이 함정을 자동으로 감지하게 함
- 결과(A/B/C, 오른손목 단일 관절): 본인 내 ICC 0.013~0.148, mean_r 0.025~0.163 — 타인 간 대조군(mean_r 0.044)과 거의 구분이 안 됨. **정렬 방법 문제가 아니라 신호 자체(단일 관절 원속도 곡선의 점별 상관)가 약하다는 뜻** — `direction_brief.md`의 기존 진단(ICC 0.05~0.07)이 정렬 버그와 무관하게 사실상 재확인된 셈
- 다음 방향은 정렬을 더 만지는 게 아니라 **특징을 바꾸는 것**: `dynamics_layer.md` 7장이 이미 제안한 대로 원속도 곡선 대신 피크 타이밍·크기, 저크 분산 같은 저차원 스칼라 특징으로 ICC를 내는 방법을 시도. 지금 방식(점별 곡선 상관)은 노이즈에 너무 취약함

## 3. 관절 수 확장 (3개 → 8개) — 완료
`pose_extract.LANDMARK_INDEX`를 오른팔 3개(손목·팔꿈치·어깨) → 양팔 6개 + 양쪽 골반 2개 = 8개로 확장. `dynamics_layer.md`의 "댄스·안무 핵심 관절: 손목·팔꿈치·어깨·골반"과 일치, scope.md 6~8개 스펙도 충족.
- `test_dynamic_id.py`(8관절 평균 거리) 재실행 통과, `analyze.py`(단일 관절만 사용, 영향 없음)도 회귀 없음 확인
- 아직 안 한 것: `analyze.py`/`homeostasis_image.py`는 여전히 `JOINT = "오른손목"` 단일 관절만 씀 — 8관절 평균을 실제로 rank-1/ICC에 반영하려면 `dynamic_id.py`/`analyze_youtube_dance.py`처럼 관절 평균 방식으로 바꿔야 함

## 4. MediaPipe Hand Landmarker — 붙여서 측정함, 전신 댄스 거리엔 채택 안 함
`pose_extract.py`에 `HandLandmarker`(모델: `models/hand_landmarker.task`, 구글 공식 저장소에서 다운로드)를 붙이고 `probe_hand_reliability()`로 검지 끝 검출률·프레임 간 떨림을 오른손목(기존 파이프라인)과 나란히 측정했다. 결과가 촬영 거리에 따라 완전히 갈렸다:
- **가까운 샷(로컬 A/B 샘플, 상반신)**: 검출률 100%, 떨림도 손목(0.0011~0.0014)보다 오히려 낮음(0.0008~0.0009) — 손가락이 더 안정적
- **전신 댄스 챌린지(`sample_dance/`)**: 검출률 4~44%(손목은 99~100%), 검출돼도 떨림이 손목의 2~4배(예: 0.047 vs 0.013) — `papers.md` G2/G3가 예상한 그대로
- **결론**: 지금 MVP의 촬영 조건(전신 댄스)에서는 손가락 keypoint가 안 쓸 만하다 — `LANDMARK_INDEX`에 편입하지 않음. 상반신 클로즈업 촬영(연기·제스처 콘텐츠, 한우진 배우 데이터 등)으로 방향이 바뀌면 재검토
- `models/hand_landmarker.task`도 `*.task`라 git에 안 잡힘 — 1번 항목의 "모델 파일 공유" 문제에 이 파일도 추가됨

## 5. 표본 확대 + Phase 0 데이터 거버넌스 (plan.md Phase 0~1)
직접 촬영 8~12명·인당 5회, 유튜브는 라이선스 확인된 채널만 우선 파일럿 3~5개 — 아직 저장소엔 3명(A/B/C) 수준 샘플뿐.
- `sample/` 확대는 촬영·동의서 정리가 먼저(성진구 담당 영역, 코드 작업 아님)
- 유튜브 소스는 라이선스 기준 정리 전까지 `sample_dance/` 확장하지 않기

## 6. 잡다한 정리 — 완료
- `requirements.txt` 추가(현재 설치된 버전 고정: opencv-python 4.13.0, numpy 2.4.6, scipy 1.18.0, matplotlib 3.11.0, mediapipe 0.10.35)
- `homeostasis_image.py`에 `analyze.py`와 동일한 `--people "이름:파일1,파일2"` 형식 CLI 추가, `PEOPLE` 하드코딩은 기본값으로만 남김. `--out`으로 저장 경로도 지정 가능
- 재실행 중 발견: 로컬 A/B(루트, 7~8초 클립)로 돌리면 ICC=0.44(A) vs 타인 대조군 0.19 — sample/compressed(3~4초, 더 짧음)보다 뚜렷한 분리가 나온다. 클립 길이·촬영 조건이 항상성 신호에 영향을 준다는 뜻일 수 있음 — 5번(표본 확대) 할 때 참고
