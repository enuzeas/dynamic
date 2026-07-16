# 쌤쌤 (구 BIO-IP — Dynamic Layer)

사람의 "움직이는 방식(스타일)"만 뽑아내 디지털 트윈·아바타·캐릭터에 입히는 프로젝트. 원래는 동작의 "콘텐츠"가 아니라 "방식"으로 개인을 식별할 수 있는가(BIO-IP)를 검증하는 것에서 출발했고, 그 실험 결과를 바탕으로 "쌤쌤 — 몸짓 복사기"로 주제를 전환했다.

## 업데이트된 테제 — 쌤쌤

**한 문장:** 사람의 움직이는 방식을 고유의 데이터로 뽑아내 살아있는 캐릭터를 만든다.

원래 가설("움직임으로 사람을 식별한다 → 생체 자산화")을 유튜브 안무 챌린지 5종·영상 50여 개로 실험한 결과 세 가지를 확인했다:

1. 같은 춤을 춰도 사람마다 다이내믹스 수치가 확실히 갈린다 — 개인차는 실재한다.
2. 다만 같은 사람도 상황(촬영 맥락)이 바뀌면 수치가 흩어진다 — "사람"뿐 아니라 "사람 + 그날의 상황"이 같이 잡혀서, 식별 수준의 항상성은 아직 증명되지 않았다.
3. (뜻밖의 발견) AI 생성 의심 영상들은 서로 움직임이 뭉치고, 실제 사람들은 제각각 넓게 퍼진다 — 도구가 겉모습이 아니라 움직임 자체를 재고 있다는 신호.

여기에 더해 "움직이는 방식"을 소유·거래할 법적 근거가 아직 없다는 법 벽까지 겹쳐, **"당신인지 알아맞히기"(식별) 대신 "당신의 움직임 느낌을 뽑아 캐릭터에 입히기"(스타일 전이)**로 방향을 바꿨다. 통계적 증명이 아니라 눈에 보이는 결과물이 기준이 되고, 창작자(게임·애니·버추얼 캐릭터·광고)라는 명확한 수요가 있으며, 필요한 기술은 전부 기존 오픈소스를 조립하면 된다.

- 파이프라인: MediaPipe 2D pose → 3D 모션(SMPL-X) → 스타일 추출·주입(Motion Puzzle) → 리타겟팅(Mixamo) → 웹 뷰어(three.js)
- 지금까지의 포즈 추출·DTW/ICC 비교 도구(`src/`)는 폐기되지 않고, "캐릭터에 입힌 움직임이 진짜 그 사람 느낌인지" 판정하는 심판 역할로 그대로 재사용된다.
- 약속하는 것: 리듬·보폭·무게중심 같은 큰 움직임 느낌의 전이. 약속하지 않는 것(지금은): 손가락 세부 동작, 미묘한 표정까지의 완벽 재현 — 멀티캠이 필요한 영역이라 범위 밖.

전환 근거 원문: [doc/team_progress_report.md](doc/team_progress_report.md)(5분 요약), [doc/samsam_worksheet.md](doc/samsam_worksheet.md)(확정 워크시트), [doc/motion_value_assessment.md](doc/motion_value_assessment.md)(판단 근거).

## 이전 가설 — BIO-IP 식별 명제 (실험적 근거)

- **고유성 (Uniqueness)** — 같은 동작을 수행한 서로 다른 사람들의 다이내믹스 서명은 통계적으로 구별된다.
- **항상성 (Permanence)** — 같은 사람의 다이내믹스 서명은 시점·맥락이 바뀌어도 일관되게 유지된다.

MediaPipe Pose로 관절 좌표를 추출하고, 속도·가속도 특징 + DTW/ICC 같은 생체인증 지표(rank-1 accuracy, EER)로 두 명제를 검증했다. 위 실험 ②에서 항상성이 아직 증명되지 않아 쌤쌤으로 전환했지만, 배경·설계 자체는 여전히 [doc/thesis.md](doc/thesis.md), [doc/scope.md](doc/scope.md)에 남아 있다.

## 구조

```
src/          분석 파이프라인 — MediaPipe pose 추출, DTW, 항상성/고유성 검증, dynamic ID 등록·식별
sample/       샘플 촬영 영상 (original/ 원본, compressed/ 압축본)
sample_dance/ 안무 챌린지 참고 영상 (로컬 전용, git 미추적)
reports/      영상별 분석 리포트 + 뷰어(viewer.html)
doc/          기획·연구 문서, 발표 자료, 분석 결과 이미지/PDF
index.html    등  Vercel 배포 페이지 (BIO-IP MVP 소개, 진행 현황 — 쌤쌤 전환 이전 버전)
```

## 설치

```bash
pip install -r src/requirements.txt
```

MediaPipe pose 모델 파일은 `.gitignore`(`*.task`)에 걸려 저장소에 포함되지 않는다. 아래 파일을 받아 직접 놓아야 한다.

- `src/models/pose_landmarker_lite.task` — [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) lite 모델
- `src/models/hand_landmarker.task` — [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) 모델 (손 신뢰도 실험용, 선택)

## 실행

스크립트는 모두 repo 루트에서 실행하는 것을 전제로 상대경로(`sample/...`)를 참조한다.

```bash
python src/demo.py --a sample/original/A_1.mp4 sample/original/A_2.mp4 sample/original/A_3.mp4 \
                   --b sample/original/B_1.mp4 sample/original/B_2.mp4 sample/original/B_3.mp4 \
                   --names "한우진" "박세준"
```

각 파이프라인 모듈은 자체 검증 스크립트를 포함한다(`src/test_*.py`, 마찬가지로 repo 루트에서 실행).

```bash
python src/test_pose_extract.py
python src/test_dynamic_id.py
python src/test_eulerian_magnify.py
python src/test_homeostasis_image.py
```

## 더 읽을거리

- [doc/team_progress_report.md](doc/team_progress_report.md) — 지금까지 온 길과 쌤쌤 전환 이유 (5분 요약)
- [doc/samsam_worksheet.md](doc/samsam_worksheet.md) — 쌤쌤 주제·역할·6회차 기술 파이프라인 확정 워크시트
- [doc/motion_value_assessment.md](doc/motion_value_assessment.md) — 스타일 전이 기술 선택 판단 근거 (Motion Puzzle, FootMR 등)
- [doc/scope.md](doc/scope.md) — (이전 가설) 구현 범위 3개월 MVP vs 5개월 전체
- [doc/thesis.md](doc/thesis.md) — (이전 가설) 핵심 명제와 검증 설계
- [doc/plan.md](doc/plan.md) — (이전 가설) 원천데이터 실행 계획
- [doc/list.md](doc/list.md) — (이전 가설) 프로젝트 개요 — 문제·솔루션·팀
