# 쌤쌤 기술 개발 스펙 — 실제 개발 문서 (2026-07-23)

> 대상 독자: 코드를 실제로 짤 사람(박세준 단독). `samsam_plan.md`가 "세션별로 언제 무엇을"이라면, 이 문서는 "어떤 파일에 어떤 함수·포맷으로"를 정한다. 기존 코드베이스(`pose_extract.py`·`analyze.py`·`demo.py`·`dynamic_id.py`, `external/motion_puzzle`)를 실제로 열어 확인한 내용 기반 — 추측 없음.

---

## 0. 핵심 발견 — 기존 코드는 "다른 트랙"이다

루트의 `pose_extract.py`·`analyze.py`·`demo.py`·`dynamic_id.py`는 전부 **다이내믹스 서명(식별) 트랙**용이다:
- `extract_joint_dynamics()`(`pose_extract.py`)는 8개 관절(`LANDMARK_INDEX`)의 2D 위치 → 속도/가속도/저크 시계열만 뽑는다. 관절 계층 구조·회전값·33개 전체 랜드마크는 없음.
- `analyze.py`/`dynamic_id.py`는 그 속도 곡선을 DTW로 비교해 "같은 사람인가"를 판정한다(rank-1 accuracy, EER).

**쌤쌤 스타일 전이(Motion Puzzle)는 이걸 그대로 못 쓴다.** `external/motion_puzzle/test.py`를 직접 확인한 결과 `--content`/`--style` 인자로 **BVH 파일**(관절 계층+회전, CMU 컨벤션, 60fps)만 받는다. 즉 "8개 관절 속도 곡선"이 아니라 "전신 관절의 3D 위치+회전을 가진 BVH 시퀀스"가 필요 — **새 추출·변환 경로가 필요하다.**

**재사용 가능한 것**: `analyze.py`의 `dtw()`(세션 6-2 평가), `demo.py`의 `trim_motion()`(길이 정렬). 그 외는 신규.

---

## 1. 실제로 새로 필요한 변환 단계 — 계획 문서에 없던 지점

`samsam_plan.md` 세션 4는 "SMPL→Mixamo" 한 단계로만 적혀 있는데, 실제로는 **두 개의 다른 리타겟**이 필요하다 — Motion Puzzle이 BVH/CMU를, three.js 뷰어가 Mixamo를 각각 요구하기 때문:

```
GVHMR 출력(SMPL-X npz)
   │
   ├─ (4-0a) hmr4d_to_npz.py → SMPL 24관절 축각 npz
   │            │
   │            └─ (4-0b, 신규 커스텀 스크립트) → CMU-21관절 리타겟 → Motion Puzzle --content 입력
   │                                                        │
   │                                              [스타일 주입] → styled BVH 출력
   │                                                        │
   └─ (4-1) ─────────────────────────────────────→ styled BVH → Mixamo 리타겟(Aberman) → three.js 뷰어
```

### 4-0a 최종 결정 (2026-07-25, smpl2bvh 실제 clone·실행 후 뒤집음)

**원래 계획(2026-07-23, 아래 "폐기된 경로" 절 참고)은 smpl2bvh로 "SMPL 24관절 BVH"를 만드는 것이었다. 실제로 clone해서 돌려본 결과 이 단계 자체가 불필요하다는 게 드러나 계획을 바꿨다:**

1. **FootMR 소스를 직접 읽어 GVHMR 출력의 실제 스키마를 확인함**(`external/FootMR/hmr4d/model/footmr/pipeline/footmr_pipeline.py`, `footmr_pl_demo.py`) — `hmr4d_results.pt`의 `smpl_params_global` 딕셔너리는 `global_orient`(F,3)·`body_pose`(F,63 = 21관절 축각)·`transl`(F,3)이고, 이 21관절 순서가 `retarget_smpl_to_cmu.py`의 `SMPL_JOINT_NAMES[1:22]`와 정확히 같은 표준 SMPL 순서다. 즉 GVHMR/FootMR 출력은 **이미 로컬 축각 회전**이라 forward kinematics도 SMPL 본체 파일(라이선스 걸림)도 필요 없이 그대로 이어붙이면 4-0b 입력이 완성된다.
2. **smpl2bvh를 실제로 clone·실행해 검증하는 과정에서 그쪽 자체의 버그를 발견함** — `mcmldm` conda env(smplx 이미 설치됨)에서 돌리려면 chumpy가 numpy≥1.24에서 제거된 `np.bool`/`np.int` 등을 참조해 깨지는 것부터 몽키패치로 우회해야 했고, 우회 후 합성 데이터(오른쪽 팔꿈치만 굽힌 60프레임)로 실제 BVH를 뽑아보니 **회전이 정확히 0인 관절마다 NaN이 나옴**(`utils/quat.py`의 `from_axis_angle()`이 `axis = rots / angle`에서 0/0 나눗셈 — 정지 관절이 하나라도 있으면, 즉 보통 모션 대부분에서 발생). `retarget_smpl_to_cmu.py`가 쓰는 `Quaternions.from_angle_axis`(motion_puzzle 자산)는 이미 0벡터를 안전 처리해서 이 문제가 없다.

**결론**: smpl2bvh clone은 `external/smpl2bvh`에 참고·시각 디버그용으로만 남겨두고(파이프라인 제외), 4-0a는 **`hmr4d_to_npz.py`**(신규, 20줄 안팎의 순수 배열 변환 — `global_orient`+`body_pose`를 이어붙이고 손(SMPL 22·23, GVHMR 출력에 아예 없음)은 항등회전, `transl`은 ×100해서 CMU BVH의 cm 스케일에 맞춤)로 확정. SMPL 모델 파일도, smplx 패키지도, 별도 env도 필요 없어져 "여전히 미결"이었던 4-0a용 env 문제 자체가 사라짐.

### 폐기된 경로 (2026-07-23 원래 계획, 기록용)

**왜 이게 단순 포맷 변환이 아니라고 봤는지**: `preprocess/generate_dataset.py`의 `process_data()`가 관절을 **이름이 아니라 고정 배열 인덱스**로 골라낸다(`global_xforms[:, np.array([0,2,3,4,5,7,8,9,10,...])]`) — 즉 입력 BVH가 CMU 원본 모캡과 **정확히 같은 관절 순서·개수**여야 한다는 점은 4-0b(CMU 리타겟)에는 여전히 유효한 발견이다. 다만 그 앞단(SMPL-X→SMPL 24관절)에 별도 변환 도구가 필요하다고 본 것이 틀렸다.

- **[KosukeFukazawa/smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh)** (MIT, 실존 확인, clone 완료) — SMPL 파라미터를 BVH로 변환하는 도구지만, 위 이유로 우리 파이프라인엔 안 씀. 의존성은 `torch`·`numpy`·`smplx`·`pickle`이고 CUDA 전용 고정은 없어 Mac 로컬(CPU)에서 실행은 됨(2026-07-25 실측 확인, `mcmldm` env + chumpy/numpy 몽키패치 필요) — 나중에 SMPL 자체 스켈레톤을 눈으로 디버그하고 싶을 때만 참고.
- **[DeepMotionEditing/deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing)** (Aberman et al., BSD-2-Clause, 1.7k★, 실존 확인) — 리타게팅 네트워크는 **Mixamo 캐릭터로 사전학습**돼 있음(README에 학습/테스트 캐릭터가 Mixamo로 명시, `datasets/__init__.py`에 하드코딩). SMPL 스켈레톤에 대해선 사전학습된 게 없어 재학습 없이는 4-0에 못 씀("학습 금지" 원칙과 충돌) — 이 저장소는 여전히 **4-1(styled BVH→Mixamo)의 실제 후보**(변경 없음, Mixamo가 이 도구의 원래 서식지).
- **안심 근거**: Aberman/Li/Weng의 동일 저장소 자체가 "CMU mocap dataset의 표준 스켈레톤"을 스타일 전이 학습 데이터 포맷으로 쓰고 있다고 README에 명시 — Motion Puzzle이 요구하는 CMU-21 컨벤션이 이 연구 계보 전체가 공유하는 표준이라는 뜻.

---

## 2. 세션별 파일 계획

| 세션 | 신규 파일(제안) | 재사용 | 데이터 계약 |
|---|---|---|---|
| 2-1 | `skeleton_extract.py` (신규 — `pose_extract.py` 확장 아님, 목적이 다름) | mediapipe 초기화 코드는 참고 가능 | 영상 → 33랜드마크 × 프레임 × (x,y,z) npz |
| 2-2 | — | `external/motion_puzzle/test.py` 그대로 호출 | BVH in/out |
| 2-3 | `analyze.py`에 함수 추가 또는 `evaluate_style_spike.py` 신규 | `dtw()` | styled BVH → 관절별 시계열 → DTW |
| 3-1/3-2 | Colab 노트북 `gvhmr_footmr_colab.ipynb` (신규) | — | 영상 → SMPL-X npz (footskate 보정 포함) |
| 4-0a | **`hmr4d_to_npz.py` — 작성·검증 완료(2026-07-25)** | — (smpl2bvh는 버그·불필요로 폐기, 위 절 참고) | GVHMR/FootMR `hmr4d_results.pt`(`smpl_params_global`) → SMPL 24관절 축각 npz(`test_hmr4d_to_npz.py`로 합성 데이터 왕복 + Motion Puzzle 통합까지 검증) |
| 4-0b | **`retarget_smpl_to_cmu.py` — 작성·검증 완료(2026-07-23)** | SMPL 24관절 축각 → CMU 원본 31관절 BVH(`--content` 입력용) | 합성 데이터로 왕복 검증 + **실제 Motion Puzzle `test.py --content`에 먹여서 스타일 전이 출력까지 확인**(`test_retarget_smpl_to_cmu.py`). 21관절로 줄여 저장하면 `process_data()`의 고정 인덱스가 깨진다는 걸 실패로 먼저 확인 → 31관절 원본 구조(10개 무해 관절은 항등회전)로 재작성해서 해결 |
| 4-1 | `external/deep-motion-editing`(Aberman, clone 필요) 호출 | Mixamo 사전학습 모델 그대로 | styled CMU-BVH → Mixamo FBX/glTF |
| 5 | `samsam_viewer.html` (신규, 프로젝트 폴더 아니라 단일 파일 — 빌드 없는 정적 사이트 컨벤션 유지) | 없음 — `reports/viewer.html`은 마크다운 리포트 뷰어, `index.html`/`mvp.html`은 2D MVP라 3D 스켈레톤과 무관 | BVH(현재) → three.js `BVHLoader` + OrbitControls, 3패널 나란히. **2026-07-23 실제 작성·Playwright 스크린샷으로 렌더링 확인 완료**(`external/motion_puzzle` 오늘 재실행 결과물 사용) — 아직 Mixamo가 아니라 BVH 그대로 표시(4-1 리타겟 전 임시), 리타겟 되면 glTF로 교체 |
| 6-2 | `evaluate_icc.py` (신규 — ICC 구현이 코드베이스에 전혀 없음, 확인함) | `analyze.py`의 DTW | styled BVH 여러 개 → ICC 계수 |

---

## 3. 환경 구성

| 환경 | 용도 | 상태 |
|---|---|---|
| 로컬 `.venv` (Python 3.14, 기존 `requirements.txt`) | mediapipe 포즈 추출, DTW/ICC 평가 | 이미 있음, 그대로 사용. `skeleton_extract.py`·`evaluate_icc.py` 신규 작성 + 자체검증 통과(2026-07-23) |
| conda env `motion_puzzle` (Python 3.8.20, PyTorch 2.4.1) | 스타일 전이 추론 | **이미 존재 + 오늘 실제로 재실행해서 확인함.** `.cuda()` 하드콜 없음(이미 `torch.device(... if is_available ...)`로 패치돼 있음), CMU 샘플로 `test.py` 정상 실행·BVH 출력 확인(`external/motion_puzzle/output/dev_verify_test/`) — "pyenv vs conda 결정 필요"는 해소됨(conda, 이미 있는 걸 그대로 씀) |
| conda env `mcmldm` (Python 3.9.23) | MCM-LDM(2순위 후보) | 이미 존재, 오늘은 미실행(코어 아님) |
| deep-motion-editing용 env | 리타겟(4-1) | **미생성** — README·소스 확인상 Mac(CPU) 로컬 실행 가능해 보임(2026-07-23 확인), `motion_puzzle` env 재사용 가능한지 다음에 확인. (4-0a는 `hmr4d_to_npz.py`로 대체돼 별도 env 불필요해짐, 2026-07-25) |
| Colab 노트북 1개 | **GVHMR + FootMR만** (같은 세션에서 이어서 실행) | **완료(2026-07-23)** — `footmr_colab.ipynb`, 실제 실행 성공. 겪은 문제 5가지(python3.10 부트스트랩, chumpy 빌드, tkinter 죽은 import, vitpose-h-wholebody 별도 링크, 파일명 특수문자)와 해결책 전부 노트북·`samsam_plan.md` 3절에 반영 |

---

## 4. 남은 결정 (2026-07-23 재확인 후 갱신)

- **해소됨(2026-07-25 재확정)**: SMPL-X→BVH(CMU) 경로 — 원래 "smpl2bvh(4-0a) + 커스텀 리타겟(4-0b)"였던 계획을 실제 실행 후 뒤집어 "`hmr4d_to_npz.py`(4-0a, smpl2bvh 없이 순수 배열 변환) + 커스텀 리타겟(4-0b, 변경 없음)"로 확정 — 근거는 위 "4-0a 최종 결정" 절. 4-1도 Aberman `deep-motion-editing`(Mixamo 사전학습 확인됨)으로 구체화, 변경 없음.
- **해소됨**: pyenv vs conda — **conda, 이미 설치돼 있고 `motion_puzzle`/`mcmldm` env도 이미 존재**. 새로 설치할 것 없음.
- **해소됨(2절 반영 완료)**: 2-1(`skeleton_extract.py`)·6-2(`evaluate_icc.py`) 오늘 작성 + 자체검증 통과. 재사용 가능.
- **확인됨**: Motion Puzzle이 이 Mac에서 지금 당장 실제로 돌아간다(오늘 재실행 확인) — "로컬 실행 가능성 높음"이 아니라 사실로 확정.
- **해소됨(2026-07-23)**: 4-0b 커스텀 리타겟 스크립트 — 작성 완료, 합성 데이터로 Motion Puzzle 실제 파이프라인 통합까지 검증 통과. GVHMR 실측 데이터 나오면 바로 연결 가능한 상태.
- **해소됨(2026-07-23)**: 세션 5 three.js 뷰어 — `samsam_viewer.html` 프로토타입 작성 + Playwright 스크린샷으로 실제 렌더링 확인(BVH 직접 로드, 아직 Mixamo 리타겟 전).
- **해소됨(2026-07-25)**: 4-0a — `hmr4d_to_npz.py` 작성, 합성 `hmr4d_results.pt` 구조 데이터로 왕복 검증 + Motion Puzzle 실제 파이프라인 통합까지 검증 통과(`test_hmr4d_to_npz.py`). GVHMR 실측 데이터 나오면 바로 연결 가능.
- **여전히 미결**: deep-motion-editing(4-1) 별도 env 필요 여부 — 아직 clone 안 함.
- **여전히 미결**: Colab 노트북 착수 시점 — GVHMR·FootMR은 Google 계정 기반 Colab 실행이 필요해 이 세션에서 직접 실행은 못 함, 노트북 초안 작성은 가능.

---

*이 문서는 `samsam_development_plan.md`(전략)·`samsam_plan.md`(세션 로그)의 하위 문서 — 실제 파일·함수·포맷 단위 스펙만 담당.*
