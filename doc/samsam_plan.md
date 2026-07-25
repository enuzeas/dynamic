# 쌤쌤 기술 파이프라인 — 단계별 실행 계획 (2026-07-15, 2026-07-23 갱신)

> `samsam_worksheet.md` 4절(다음 캡스톤 기술 배분, 6회)을 오늘자 문헌 검색 결과(`motion_value_assessment.md` 11절, `papers.md` C·H절)로 구체화한 실행 계획. 판단 근거는 위 두 문서에 있고, 여기는 "누가 언제 무엇을 어떤 도구로" 만 담는다.
> 배경·목표·일정·리스크를 하나로 정리한 정식 개발기획서는 `samsam_development_plan.md` 참고 — 이 문서는 그 기획서의 세션별 상세 실행 로그.

## 0. 전제 3가지 (worksheet에서 그대로 계승)

1. **조립이지 연구가 아니다.** 전 구간 사전학습·공개 코드만 사용, 처음부터 학습 금지.
2. **캡스톤 코어는 하나.** "스타일 전이가 웹 뷰어에서 사람별로 보인다." 플랫폼·디지털트윈·광고영상은 발표 포장(기획·마케팅·콘텐츠 영역)이며 기술개발 시간을 먹지 않는다.
3. **성공 기준은 소박하게 고정.** DTW로 "A 출력이 A 원본에 붙고 B와 벌어진다" + 웹 뷰어 육안 확인. 큰 패턴(리듬·보폭·무게중심)까지만 — 손가락·감정 뉘앙스는 범위 밖.

담당: 기술개발 박세준 단독(병목), 렌더/뷰어는 송필순 주도, 데이터 수집·평가는 전원.

---

## 1. 착수 체크리스트 — 2026-07-15 확인 완료

| # | 할 일 | 결과 |
|---|---|---|
| 1 | Motion Puzzle 공식 레포 clone | **완료** — `external/motion_puzzle`. Python 3.8, PyTorch>=1.10, CUDA 특정 안 함, `.cuda()` 3곳뿐(패치 쉬움). 사전학습 가중치·CMU 테스트 BVH 샘플 다운로드 스크립트 포함(`download.sh`). |
| 1-1 | (신규) MCM-LDM(CVPR 2024, motion-to-motion diffusion) 레포 clone | **완료** — `external/MCM-LDM`. Python 3.9, PyTorch 1.12.1, `.cuda()`/`device` 32곳 — 이식 공수 큼. papers.md C14 참고. |
| 2 | FootMR(papers.md H1) 공개 코드 확인 | **완료, 그러나 로컬 실행 불가.** `github.com/twehrbein/FootMR` clone 완료(`external/FootMR`)했지만 `requirements.txt`가 `torch==2.3.0+cu121` + `pytorch3d` linux_x86_64 CUDA 휠로 고정돼 있어 M-series Mac에서 못 돈다. |
| 3 | VQ-Style(C7), Constrained Diffusion(C8) 공개 코드 확인 | **완료 — 둘 다 공개 코드 없음, 캡스톤 스코프에서 완전히 제외.** |
| 4 | AMASS 또는 Mixamo에서 스타일이 뚜렷이 다른 캐릭터/모션 2~3종 확보 | **미착수.** Motion Puzzle 저장소의 CMU 테스트 BVH 샘플로 1차 대체 가능(`download.sh datasets`) — 별도 AMASS/Mixamo 계정 없이 스파이크 시작 가능. |

**환경 관련 발견(중요, 결정 필요):**
- 현재 프로젝트 venv는 Python 3.14.6(mediapipe/opencv 전용, 경량). Motion Puzzle(3.8)·MCM-LDM(3.9)·FootMR(3.10대)은 전부 훨씬 오래된 Python/PyTorch를 요구해 지금 venv와 공존 불가 — 각각 별도 가상환경 필요(pyenv 또는 conda/miniforge, 현재 둘 다 미설치 상태).
- **FootMR은 Mac에서 원천적으로 안 돌아간다(CUDA+Linux 전용 pytorch3d 휠).** 로컬에 억지로 이식하기보다 **Google Colab(무료 T4 GPU, Linux)**에서 1회성으로 돌려 "발목 오차 감소율" 숫자만 뽑아오는 쪽을 권장 — 어차피 스파이크는 "버려도 되는 실험"이라 상시 로컬 파이프라인에 안 넣어도 됨.
- **MCM-LDM도 이식 공수·추론 속도 부담이 있어** 같은 이유로 처음엔 Colab에서 시도해보고, 잘 되면 로컬 이식을 나중에 고려하는 순서를 권장.
- **Motion Puzzle만 로컬(Mac, CPU)로 바로 시도**해볼 만함 — 패치할 `.cuda()` 호출 3곳만 `.to(device)`로 바꾸면 CPU에서 돌아갈 가능성이 높음.

→ **(a) 해소됨(2026-07-23):** conda 이미 설치돼 있고 `motion_puzzle`(Python 3.8.20, PyTorch 2.4.1)·`mcmldm`(Python 3.9.23) env도 이미 존재 — 오늘 `motion_puzzle` env에서 CMU 샘플로 `test.py` 재실행해 실제로 동작 확인함(`external/motion_puzzle/output/dev_verify_test/`). 새로 설치할 것 없음.
→ **다음 결정 필요:** (b) Colab 노트북을 지금 만들지 여부. 사용자 확인 후 진행.

---

## 2. 세션 1~2 — 2D 포즈 + 데이터 수집 + 스타일 전이 스파이크①

**목표 상태:** 영상에서 포즈가 추출된다 + Motion Puzzle 출력이 스타일별로 DTW상 구분되는지 답이 나온다 + footskate 보정 효과를 실측한다.

| 단계 | 작업 | 도구 |
|---|---|---|
| 2-1 | 영상 → 2D 포즈 추출 파이프라인 | MediaPipe Pose |
| 2-2 | Motion Puzzle을 AMASS/Mixamo 공개 모캡에 그대로 실행 — 같은 content에 스타일 A/B 주입 후 출력 비교 | Motion Puzzle (사전학습) |
| 2-3 | 2-2 출력에 DTW 적용 — A 출력↔A 원본 거리 vs A 출력↔B 원본 거리 | 기존 `analyze.py`의 DTW 함수 |
| 2-4 | (병행) 모노큘러 3D 캡처 출력에 FootMR 적용 전/후 발목 안정성 비교 | FootMR + 기존 3D 캡처 결과물 |
| 2-5 | 데이터 수집 시작 — 한 사람 × 여러 동작(걷기·뛰기·제스처·감정), 양보다 다양성 | 직접 촬영 |

**게이트 (5회차로 안 넘어가고 여기서 판정):**
- DTW상 스타일 A/B가 갈라짐 → 통과, 계획대로 5회차 진행.
- 안 갈라짐 → `samsam_worksheet.md`의 "실패 시 대비"(이미 있는 DTW 분석 자산으로 "생성 대신 분석 데모") 전환 논의를 이 시점에 시작.

**2026-07-15 스파이크 1차 실행 결과(공개 CMU 모캡, 콘텐츠=127_21 고정, 스타일 4종 비교):**
- **시각 확인 — 통과.** 스타일 4종(142_21·55_07·41_02·137_11)을 같은 콘텐츠에 입힌 출력을 3D 스켈레톤으로 겹쳐 그려보니 프레임별로 팔·다리 자세가 눈에 띄게 갈라짐 ([motion_puzzle_style_spike.png](../motion_puzzle_style_spike.png), [motion_puzzle_style_spike_4styles.png](../motion_puzzle_style_spike_4styles.png)).
- **정량 확인(출력끼리 직접 DTW 비교) — 대체로 통과, 예외 1건.** 4개 출력 쌍의 RightHand DTW 거리: 대부분 3.9~5.6로 서로 갈라지지만 **41_02↔137_11 쌍만 0.88로 거의 동일** — 이 두 스타일 소스 자체가 원래 비슷했거나, 모델이 이 쌍을 못 갈랐거나 둘 중 하나(다음 세션에서 원본 클립끼리 비교해 원인 확인 필요).
- **콘텐츠 보존 — 통과.** 4개 출력 모두 원본 콘텐츠와의 거리가 3.37~4.65로 좁은 범위에 몰림 — 스타일이 바뀌어도 "무슨 동작인지"는 안정적으로 유지됨.
- **평가 방법론 교훈:** 처음엔 출력을 스타일 "원본 전체 클립"과 DTW로 비교했는데, 원본 클립끼리 애초에 다른 종류의 동작(콘텐츠)이라 "스타일 차이"와 "동작 종류 차이"가 섞여 숫자가 뒤죽박죽이었다. **콘텐츠를 고정하고 출력끼리 직접 비교**하는 쪽이 훨씬 깨끗한 신호를 준다 — 8-2절 점수식을 쓸 때 이 교훈을 반영해 "출력 vs 원본 스타일 클립"이 아니라 "동일 콘텐츠의 출력끼리 비교"를 기본으로 삼는다.
- **종합 판정: 게이트 통과.** 5회차(우리 데이터로 재현)로 진행 가능. 단, 스타일 페어에 따라 분리도가 들쭉날쭉하므로 우리 데이터 촬영 시 "확실히 다른 스타일"로 보이는 동작 쌍을 의도적으로 고르는 게 안전.

---

## 3. 세션 3 — 2D→3D 모션 (SMPL-X) + 스파이크②

**목표 상태:** 포즈가 3D 모션이 된다 + 교수 질문 1(footskate)에 실측으로 답한다.

| 단계 | 작업 | 도구 |
|---|---|---|
| 3-1 | 2D→3D 리프팅 | **GVHMR로 결정(2026-07-23).** 근거는 아래 참고 |
| 3-2 | FootMR을 실제 파이프라인에 통합(세션 2에서 효과 확인됐다면) | **완료(2026-07-23)** — Colab에서 실제로 끝까지 실행 성공(`footmr_colab.ipynb`). 겪은 문제·해결책은 아래 참고 |
| 3-3 | 우리 데이터로 3D 모션 육안 검수 — footskate·지터 정도 기록 | 육안 확인은 됨(렌더 영상 생성 확인). 정량 수치화는 다음 과제 |

**산출물:** "단안으로 어디까지 되는가"에 대한 수치(발목 오차 감소율) — 교수 상담 때 추측 대신 이 숫자로 대화.

**3-1 결정 근거 (2026-07-23, MediaPipe 3D landmark / WHAM / GVHMR 실측 비교 결과):**
- **FootMR은 GVHMR 코드베이스를 그대로 포크·확장한 것**임을 확인 — `twehrbein/FootMR`의 `requirements.txt`가 `zju3dv/GVHMR`과 바이트 단위로 동일(`torch==2.3.0+cu121` 등)하고, 같은 체크포인트(ViTPose·YOLO·DPVO)를 요구. 즉 WHAM이나 MediaPipe를 골랐다면 FootMR이 기대하는 입력 포맷(ViTPose 2D keypoint + HMR2 피처)에 맞추는 글루 코드를 새로 짜야 해서 "조립이지 연구가 아니다" 원칙(0절)을 어기게 됨 — GVHMR을 고르면 이 문제 자체가 없음.
- 정확도도 GVHMR이 우세: 3DPW PA-MPJPE 36.2mm, world-grounded 지표(RICH WA-MPJPE₁₀₀) 78.8mm — WHAM(각각 35.9mm/109.9mm)보다 월드 궤적(=footskate 판정에 필요한 지표)에서 뚜렷이 앞섬. 핵심망 추론도 0.28초/시퀀스로 WHAM(2.0초)의 약 7배.
- GVHMR도 FootMR과 똑같이 CUDA+Linux 전용(pytorch3d 휠)이라 Mac 로컬 불가·Colab 필수 — 하지만 FootMR을 쓰기로 한 순간 이미 Colab은 확정 비용이었으므로, **하나의 Colab 세션에서 GVHMR→FootMR을 이어서 실행**하는 게 오히려 환경을 두 번 만드는 것보다 쌈.
- MediaPipe 3D world landmark(로컬 즉시 무료)는 완전히 버리지 않음 — world-grounding이 없어 footskate 판정엔 원천적으로 약하지만, Motion Puzzle 입력 실험 등 footskate와 무관한 용도로는 여전히 씀.
- 라이선스 참고: GVHMR은 비상업 연구·교육용(수정판도 오픈소스 의무) — 캡스톤 용도엔 문제없음, 상업화 시엔 별도 확인 필요.

**Colab 실행 중 실제로 겪은 문제 5가지 (2026-07-23, `footmr_colab.ipynb`에 전부 반영 완료)**:
1. **Colab 기본 Python이 3.12로 올라가 있어 pytorch3d의 cp310 전용 사전빌드 휠과 안 맞음** → apt로 python3.10 설치 + `get-pip.py`로 직접 부트스트랩(venv는 apt 인덱스 문제로 실패해서 포기).
2. **chumpy가 pip 빌드 격리 환경에 numpy가 없어서 빌드 실패** → numpy 먼저 설치 후 `--no-build-isolation`으로 우회.
3. **FootMR 소스 자체의 죽은 import**(`body_model.py`의 `from turtle import forward`, 아무 기능도 안 씀)가 `tkinter` 요구 → `python3.10-tk` 설치로 우회.
4. **`vitpose-h-wholebody.pth`가 GVHMR 체크포인트 폴더에 없음** — FootMR 전용 Nextcloud 링크에만 있었음(`footmr_checkpoint.ckpt`와 같은 곳).
5. **입력 영상 파일명에 특수문자(`()`, `@`)가 있으면 셸도 Hydra의 config override 파서도 깨짐** → 업로드 직후 안전한 파일명으로 강제 rename.

**추가 확인 필요**: 기본 후처리(`no_postproc=False`)가 미세한 타이밍·관절 디테일을 누르는 것으로 보임(demo.py 자체 문서화) — 쌤쌤 코어가 "그 사람 특유의 디테일" 보존이 목적이라, 세션 6 실데이터 평가 때 `--no_postproc` 버전도 같이 비교해서 어느 쪽이 개인 스타일 신호를 더 잘 보존하는지 확인할 것.

---

## 4. 세션 4 — 리타겟팅 (실제로는 2단계, 2026-07-23 재확인)

**중요 정정**: "SMPL→Mixamo" 한 단계가 아니라, Motion Puzzle(BVH/CMU 요구)과 three.js 뷰어(Mixamo 요구)가 서로 다른 스켈레톤을 요구해서 **리타겟이 앞뒤로 한 번씩 총 2번** 필요하다. 상세 근거·확인한 도구는 `samsam_dev_spec.md` 1절 참고.

| 단계 | 작업 | 도구 |
|---|---|---|
| 4-0a | GVHMR SMPL-X 출력 → SMPL 24관절 회전 | **완료(2026-07-25)** — `hmr4d_to_npz.py`. 원래 계획한 `smpl2bvh`는 clone·실행까지 해봤으나 (a) GVHMR 출력이 이미 로컬 축각이라 변환기 자체가 불필요했고 (b) 실행해보니 0-회전 관절에서 NaN 나는 버그 발견 — 둘 다 실제 실행으로 확인, 근거는 `samsam_dev_spec.md` 1절 |
| 4-0b | SMPL 24관절 → CMU 원본 31관절 BVH 리타겟(Motion Puzzle 입력용) | **완료(2026-07-23)** — `retarget_smpl_to_cmu.py` 작성, 합성 데이터로 실제 `test.py --content`에 먹여 스타일 전이 출력까지 확인(`test_retarget_smpl_to_cmu.py`). 2026-07-25: `hmr4d_to_npz.py` 출력을 입력으로 한 통합 검증도 통과(`test_hmr4d_to_npz.py`) |
| 4-1 | styled CMU-BVH → Mixamo 리타겟(뷰어용) | `deep-motion-editing`(Aberman et al., BSD-2-Clause, Mixamo 사전학습 확인됨, clone 필요) |
| 4-2 | footskate·관통(penetration) 발생 시에만 업그레이드 검토 | papers.md H4 (Spatially Adaptive Interaction Guidance) — 코드 있을 때만, 없으면 스킵 |

모션이 캐릭터에 입혀지는 게 목표 상태. 문제 없으면 4-2는 건너뛴다(조기 최적화 금지). **4-0b는 clone-and-run이 아니라 실제 코딩 작업이라 다른 단계보다 시간이 더 걸릴 수 있음** — 세션 일정 잡을 때 반영.

---

## 5. 세션 5 — 렌더/뷰어 (three.js)

담당: **송필순 주도**, 박세준은 모션 데이터 포맷만 전달(1인 개발 병목 완화).

| 단계 | 작업 | 도구 |
|---|---|---|
| 5-1 | Mixamo 캐릭터 + three.js 뷰어에 스타일 전이 전/후 나란히 재생 | three.js |
| 5-2 | 웹에서 before/after를 눈으로 비교 가능하게 | — |

목표 상태: 웹에서 before/after를 본다.

---

## 6. 세션 6 — 우리 데이터로 style 추출·주입 + 평가 + 통합

| 단계 | 작업 | 도구 |
|---|---|---|
| 6-1 | Motion Puzzle을 세션 2-5에서 확보한 우리 데이터(한 사람 × 여러 동작)에 적용 | Motion Puzzle |
| 6-2 | DTW·ICC로 정량 평가 | 기존 `analyze.py` |
| 6-3 | 플랫폼 통합, 발표 데모 완성 | 전원 |

목표 상태: 같은 동작이 사람마다 다른 결로 움직인다 — 완성 데모.

---

## 7 (후보). 세션 7 — 카메라 프리비즈 → AI 영상 생성 (조건부, 코어 아님)

세션 6까지 완료된 뒤 팀이 편입 여부를 별도 결정. 상세 계획·모델 비교(MusePose 1순위)는 `samsam_camera_previs_plan.md` 참고. 코어(세션 1-6)와 독립적이라 실패해도 코어 완성 데모에는 영향 없음.

---

## 8. 전체를 관통하는 리스크 게이트 요약

| 시점 | 질문 | Yes | No |
|---|---|---|---|
| 세션 1~2 | Motion Puzzle 출력이 스타일별로 DTW상 갈라지나? | 5회차까지 계획대로 진행 | "분석 데모"로 스코프 축소 논의 |
| 세션 1~2 | FootMR이 footskate를 실측으로 줄이나? | 세션 3에 통합 | 멀티캠 필요성을 교수님과 재상담(질문 1 재오픈), Plan B는 `papers.md` H5(freemocap) |
| 세션 1~2 | VQ-Style/Constrained Diffusion 공개 코드가 있나? | 곁다리 후보로 세션 6 이후 검토 | 완전 제외, Motion Puzzle 유지 |
| 세션 4 | 리타겟팅에서 footskate/관통이 실제로 보이나? | H4 코드 유무 확인 후 도입 검토 | 표준 리타겟 그대로 유지 |
| 세션 7(후보) | 카메라 각도 변화가 AI 생성 결과에 육안으로 반영되나? | "세션 7" 정식 편입 논의 | 프리비즈 영상은 감독용 참고자료로만 사용, 자동 파이프라인 폐기 |
