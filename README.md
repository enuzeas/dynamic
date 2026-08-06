# 쌤쌤 (구 BIO-IP — Dynamic Layer)

사람의 "움직이는 방식(스타일)"만 뽑아내 디지털 트윈·아바타·캐릭터에 입히는 프로젝트. 원래는 동작의 "콘텐츠"가 아니라 "방식"으로 개인을 식별할 수 있는가(BIO-IP)를 검증하는 것에서 출발했고, 그 실험 결과를 바탕으로 "쌤쌤 — 몸짓 복사기"로 주제를 전환했다.

---

## 1. 컨셉

**한 문장:** 사람의 움직이는 방식을 고유의 데이터로 뽑아내 살아있는 캐릭터를 만든다.

원래 가설("움직임으로 사람을 식별한다 → 생체 자산화")을 유튜브 안무 챌린지 5종·영상 50여 개로 실험한 결과 세 가지를 확인했다:

1. 같은 춤을 춰도 사람마다 다이내믹스 수치가 확실히 갈린다 — 개인차는 실재한다.
2. 다만 같은 사람도 상황(촬영 맥락)이 바뀌면 수치가 흩어진다 — "사람"뿐 아니라 "사람 + 그날의 상황"이 같이 잡혀서, 식별 수준의 항상성은 아직 증명되지 않았다.
3. (뜻밖의 발견) AI 생성 의심 영상들은 서로 움직임이 뭉치고, 실제 사람들은 제각각 넓게 퍼진다 — 도구가 겉모습이 아니라 움직임 자체를 재고 있다는 신호.

여기에 "움직이는 방식"을 소유·거래할 법적 근거가 아직 없다는 벽까지 겹쳐, **"당신인지 알아맞히기"(식별) 대신 "당신의 움직임 느낌을 뽑아 캐릭터에 입히기"(스타일 전이)**로 방향을 바꿨다. 통계적 증명이 아니라 눈에 보이는 결과물이 기준이 되고, 창작자(게임·애니·버추얼 캐릭터·광고)라는 명확한 수요가 있으며, 필요한 기술은 새로 연구하지 않고 **검증된 오픈소스를 조립**하면 된다.

- 약속하는 것: 리듬·보폭·무게중심 같은 큰 움직임 느낌의 전이.
- 약속하지 않는 것(지금은): 손가락 세부 동작까지의 완벽 재현(멀티캠 필요), 표정(2026-08-05 자문 결과 몸짓에 집중하기로 하며 보류 — 6장 참고).

전환 근거 원문: [doc/samsam/team_progress_report.md](doc/samsam/team_progress_report.md)(5분 요약), [doc/samsam/samsam_worksheet.md](doc/samsam/samsam_worksheet.md)(확정 워크시트), [doc/bio-ip-archive/motion_value_assessment.md](doc/bio-ip-archive/motion_value_assessment.md)(판단 근거).

---

## 2. 지금까지 정리된 내용 — 핵심 기술 결정

새 알고리즘을 발명하지 않고, 이미 공개·검증된 기술을 이 프로젝트 데이터에 맞게 조립한다는 원칙 아래 다음을 확정했다.

| 구성요소 | 채택 | 근거 |
|---|---|---|
| 영상 → 3D 모션 | **GVHMR + FootMR** (Colab T4) | MediaPipe 3D landmark보다 world-grounded 지표 우세, WHAM보다 추론 7배 빠름 |
| 스타일 추출·전송 (몸짓) | **Motion Puzzle** (Jang et al. 2022) | 신체 부위별(5부위) AdaIN+attention, 라벨·페어링 없이 임의 스타일 전송 가능. 로컬(Mac)에서 실행 확인 완료 |
| 개인 고유 특징 정제 | Motion Puzzle `Encoder_sty` 위에 **Triplet Loss(Batch Hard, P명×K클립)** 정제 헤드 신규 학습 | 구현 후보 `pytorch-metric-learning`으로 확정. ⚠ 다만 2026-08-05 원저자 자문 결과 이 설계의 전제(여러 동작 걸친 개인 불변 성분 = 사실상 "스타일"이 아니라 "습관")가 재검토 대상이 됨 |
| 리타겟팅 | SMPL→CMU-BVH(Motion Puzzle 입력용) + CMU→Mixamo(뷰어용), 2단 리타겟 | 전 관절 방향 검증 완료 |
| 정량 평가 | DTW(재현 신뢰도) + ICC(반복 촬영 간 일관성) + LOMO(동작 하나 빼고 재식별) | 이미 구현된 도구(`src/analyze.py`, `evaluate_icc.py`) 재사용. ICC+LOMO 조합은 2026-08-05 원저자에게 "개인 고유성 검증엔 적절한 방법"이라고 확인받음 |
| 표정 스타일 (병행 트랙) | **보류(2026-08-05)** | 얼굴은 몸짓과 다른 영역이고 자문 교수 랩 소관도 아니라, 하나에 집중하라는 권고를 받아 몸짓에 집중하기로 정리. 조사 내용은 `motion_ip_pipeline.md` 8장에 참고용으로 보존 |

상세 아키텍처·손실 함수·PMSR 정의는 [doc/samsam/motion_ip_pipeline.md](doc/samsam/motion_ip_pipeline.md) 참고(1-1장에 자문 결과 정리). `professor_review.md`(원 기획서) 항목별 실제 구현 현황 대조는 [doc/samsam/professor_review_status.md](doc/samsam/professor_review_status.md)에 정리돼 있다.

---

## 3. 진행 상황

### 완료

- 영상 → 2D/3D 포즈 추정 (GVHMR+FootMR, Colab)
- 신체 부위별 스타일 인코딩·전송 (Motion Puzzle, 로컬 실행 확인)
- 2단 리타겟팅 (SMPL→CMU-BVH, CMU→Mixamo)
- 웹 뷰어 — 원본/스타일A/스타일B 3패널 비교 데모 (`samsam_viewer.html`, `style_transfer.html`)
- DTW·ICC 정량 평가 도구 (도구는 완성, 아래 항목에 배선만 남음)
- **Motion Puzzle 원저자(이성희 교수, KAIST CT) 자문 완료(2026-08-05)** — 담당 대학원생 배정, 정기 협업 구조 확정. 회의록: `External_doc/20260805_이성희교수_회의록.md`

### 앞으로 진행해야 할 부분

1. **개인 고유 특징 정제 접근 재검토(최우선)** — 자문 결과 현재 설계(Encoder_sty + Triplet Loss)의 전제가 "스타일"이 아니라 "습관" 문제일 수 있다는 지적을 받음. 스코프를 좁힐지, 그대로 밀지, 개인 식별 분류기 쪽으로 무게중심을 옮길지 대학원생과 함께 결정 필요.
2. **1번 결정 후 정제 헤드 구현** — 구현 후보(`pytorch-metric-learning`) 조사는 끝남, 접근 방식만 확정되면 바로 착수 가능.
3. **ICC·DTW·LOMO를 정제된 임베딩에 배선** — 반복 촬영 간 일관성(ICC), 원본-재현 유사도(DTW), 동작 하나 뺀 재식별(LOMO) 검증을 위 결과에 연결.
4. **영상 기반 개인화 아바타 생성(외형)** — 아직 미착수. 지금 뷰어는 고정 Mixamo 캐릭터만 사용.
5. **개인 IP 임베딩 스토어** — 식별자 기반 저장·버전 관리. 1번이 안정화된 뒤 순서상 자연스러움.

표정 스타일 트랙은 2026-08-05 자문 결과 보류(위 2장 표 참고) — 이 목록에서 제외했다. 항목별 상세 근거는 [doc/samsam/professor_review_status.md](doc/samsam/professor_review_status.md) 참고.

---

## 4. 지원이 필요한 부분

- **안정적인 GPU 자원.** 영상→3D 모션(GVHMR+FootMR) 단계는 GPU가 구조적으로 필수인데, 현재 무료 Colab T4 세션에 의존 중이라 세션 중 GPU 미배정·끊김 변동성이 실전 리스크다. 정제 헤드를 대규모 공개 데이터(NTU RGB+D 등)로 사전학습하기로 결정되면 그 학습에도 GPU가 필요하다 (`doc/samsam/samsam_gpu_requirements.md`).
- **데이터 수집 인력·일정.** 15~20명 × 6~8동작(캘리브레이션 포함) × 3테이크, 1인당 약 15~18분. 재촬영(S2, 지문 검증의 핵심) 대상자 5명 확보도 필요.
- **개인 고유 특징 정제 접근 확정.** 원저자 자문 결과 현재 설계 전제가 재검토 대상이 됨 — 대학원생과의 첫 정기 미팅에서 스코프(동작 내 개인차 vs 습관 추출 vs 식별 분류기)를 확정하는 게 다음 병목.
- **권리·동의 인프라 설계.** 파이프라인이 완성되면 "서명을 훔쳐 다른 사람인 척 재현"도 기술적으로 가능해지므로, 동의·워터마킹 같은 권리 인프라를 파이프라인 구현과 병행 또는 선행해서 설계할 필요가 있다. (2026-08-05 자문에서도 개인 습관적 동작 자체의 법적 보호는 불분명하다는 의견을 받음 — `External_doc/20260805_이성희교수_회의록.md`)

> Motion Puzzle 원저자(KAIST 이성희 교수) 자문·협업은 2026-08-05 확보 완료 — 담당 대학원생이 배정돼 정기 미팅으로 이어간다.

---

## 5. 이전 가설 — BIO-IP 식별 명제 (실험적 근거)

- **고유성 (Uniqueness)** — 같은 동작을 수행한 서로 다른 사람들의 다이내믹스 서명은 통계적으로 구별된다.
- **항상성 (Permanence)** — 같은 사람의 다이내믹스 서명은 시점·맥락이 바뀌어도 일관되게 유지된다.

MediaPipe Pose로 관절 좌표를 추출하고, 속도·가속도 특징 + DTW/ICC 같은 생체인증 지표(rank-1 accuracy, EER)로 두 명제를 검증했다. 항상성이 아직 증명되지 않아 쌤쌤으로 전환했지만, 배경·설계 자체는 여전히 [doc/bio-ip-archive/thesis.md](doc/bio-ip-archive/thesis.md), [doc/bio-ip-archive/scope.md](doc/bio-ip-archive/scope.md)에 남아 있다.

---

## 6. 구조

```
src/                    분석 파이프라인 — MediaPipe pose 추출, DTW, 항상성/고유성 검증, dynamic ID 등록·식별
sample/                 샘플 촬영 영상 (original/ 원본, compressed/ 압축본)
reports/                영상별 분석 리포트 + 뷰어(viewer.html)
doc/samsam/             쌤쌤(현재 트랙) — 파이프라인 아키텍처, 세션별 실행 계획, 촬영 가이드, GPU 산정, 발표 자료
doc/bio-ip-archive/     BIO-IP(이전 가설) — 식별 명제 검증 실험, 안무 챌린지 분석, 배경 문서
External_doc/           팀원 별도 작성 자료 — 몸짓 촬영 체크리스트 v2, 표정 스타일 트랙 조사(보류), 이성희 교수 자문 준비·회의록
evaluate_icc.py, hmr4d_to_npz.py, retarget_smpl_to_cmu.py,
skeleton_extract.py, style_transfer_server.py            파이프라인 스크립트 (repo 루트에서 실행 전제, external/motion_puzzle 상대경로 참조)
index.html, report.html, samsam_viewer.html, style_transfer.html   Vercel 배포 페이지 + 인터랙티브 데모
```

---

## 7. 설치

```bash
pip install -r src/requirements.txt
```

MediaPipe pose 모델 파일은 `.gitignore`(`*.task`)에 걸려 저장소에 포함되지 않는다. 아래 파일을 받아 직접 놓아야 한다.

- `src/models/pose_landmarker_lite.task` — [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) lite 모델
- `src/models/hand_landmarker.task` — [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) 모델 (손 신뢰도 실험용, 선택)

스타일 전이 데모(`style_transfer_server.py`)는 별도로 conda env `motion_puzzle`(Python 3.8, PyTorch 2.4.1)이 필요하다 — `external/motion_puzzle`의 README 참고.

## 8. 실행

스크립트는 모두 repo 루트에서 실행하는 것을 전제로 상대경로를 참조한다.

```bash
python src/demo.py --a sample/original/A_1.mp4 sample/original/A_2.mp4 sample/original/A_3.mp4 \
                   --b sample/original/B_1.mp4 sample/original/B_2.mp4 sample/original/B_3.mp4 \
                   --names "한우진" "박세준"
```

각 파이프라인 모듈은 자체 검증 스크립트를 포함한다(`src/test_*.py`, `test_*.py`, 마찬가지로 repo 루트에서 실행).

```bash
python src/test_pose_extract.py
python src/test_dynamic_id.py
python src/test_eulerian_magnify.py
python src/test_homeostasis_image.py

python test_hmr4d_to_npz.py
python test_retarget_smpl_to_cmu.py
python test_evaluate_icc.py
python test_skeleton_extract.py
```

스타일 전이 인터랙티브 데모:

```bash
python style_transfer_server.py   # http://localhost:8940
```

---

## 9. 더 읽을거리

**현재 트랙(쌤쌤)**
- [doc/samsam/motion_ip_pipeline.md](doc/samsam/motion_ip_pipeline.md) — 전체 기술 아키텍처, 몸짓·표정 두 트랙의 구성요소·손실 함수·다음 단계
- [doc/samsam/professor_review_status.md](doc/samsam/professor_review_status.md) — 기획서 항목별 실제 구현 현황 대조
- [doc/samsam/team_progress_report.md](doc/samsam/team_progress_report.md) — 지금까지 온 길과 쌤쌤 전환 이유 (5분 요약)
- [doc/samsam/samsam_plan.md](doc/samsam/samsam_plan.md) — 세션별(1~7) 실행 로그
- [doc/samsam/samsam_gpu_requirements.md](doc/samsam/samsam_gpu_requirements.md) / [samsam_gpu_time_estimate.md](doc/samsam/samsam_gpu_time_estimate.md) — GPU 필요 여부·소요 시간 추정
- [External_doc/20260805_이성희교수_회의록.md](External_doc/20260805_이성희교수_회의록.md) — Motion Puzzle 원저자 자문 결과 원문(스타일-습관 구분, 검증 방법 확인, 표정 트랙 보류 결정 등)
- [External_doc/](External_doc/) — 몸짓 촬영 체크리스트 v2, 표정 스타일 트랙 조사(보류), 자문 준비 자료

**이전 가설(BIO-IP)**
- [doc/bio-ip-archive/motion_value_assessment.md](doc/bio-ip-archive/motion_value_assessment.md) — 스타일 전이 기술 선택 판단 근거 (Motion Puzzle, FootMR 등)
- [doc/bio-ip-archive/scope.md](doc/bio-ip-archive/scope.md) — 구현 범위 3개월 MVP vs 5개월 전체
- [doc/bio-ip-archive/thesis.md](doc/bio-ip-archive/thesis.md) — 핵심 명제와 검증 설계
- [doc/bio-ip-archive/plan.md](doc/bio-ip-archive/plan.md) — 원천데이터 실행 계획
- [doc/bio-ip-archive/list.md](doc/bio-ip-archive/list.md) — 프로젝트 개요 — 문제·솔루션·팀
