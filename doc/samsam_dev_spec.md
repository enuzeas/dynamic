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
   ├─ (4-0a) smpl2bvh → SMPL 자체 24관절 BVH
   │            │
   │            └─ (4-0b, 신규 커스텀 스크립트) → CMU-21관절 리타겟 → Motion Puzzle --content 입력
   │                                                        │
   │                                              [스타일 주입] → styled BVH 출력
   │                                                        │
   └─ (4-1) ─────────────────────────────────────→ styled BVH → Mixamo 리타겟(Aberman) → three.js 뷰어
```

### 4-0 재확인 결과 (2026-07-23, 실제 코드·리포 확인 — "확인 필요" 상태 해소)

**왜 이게 단순 포맷 변환이 아닌지 코드로 확인함**: `preprocess/generate_dataset.py`의 `process_data()`가 관절을 **이름이 아니라 고정 배열 인덱스**로 골라낸다(`global_xforms[:, np.array([0,2,3,4,5,7,8,9,10,...])]`) — 즉 입력 BVH가 CMU 원본 모캡과 **정확히 같은 관절 순서·개수**여야 한다. 게다가 `configs/skeleton_cmu.yaml`이 CMU 21관절의 **고정 rest-pose 오프셋**(본 길이)을 그래프 컨볼루션 구조에 박아 넣는다. 그래서 "SMPL 출력을 BVH로 저장"만으론 안 되고, **CMU의 정확한 스켈레톤 구조로 리타겟**해야 한다.

**실제 조사 결과 (gh api·README·소스코드 직접 확인)**:
- **[KosukeFukazawa/smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh)** (MIT, 실존 확인) — SMPL 파라미터를 BVH로 변환. `smpl2bvh.py` 소스를 직접 읽어보니 `--model_type` 인자가 `["smpl", "smplx"]` 둘 다 지원 — **GVHMR의 SMPL-X 출력을 별도 가공 없이 바로 받을 수 있음**(예상보다 쉬움, "SMPL 24관절로 수동 추출" 단계 불필요). 의존성은 `torch`·`numpy`·`smplx`·`pickle`뿐, CUDA 전용 고정 없음 — **Mac 로컬(CPU)에서 돌아갈 가능성 높음**(requirements.txt 없이 소스 직접 확인, GVHMR/FootMR 같은 Colab 전용 제약 없음). 단, 출력은 여전히 **SMPL(-X) 자신의 스켈레톤**이지 CMU-21이 아니라서 4-0b는 그대로 필요.
- **[DeepMotionEditing/deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing)** (Aberman et al., BSD-2-Clause, 1.7k★, 실존 확인) — 리타게팅 네트워크는 **Mixamo 캐릭터로 사전학습**돼 있음(README에 학습/테스트 캐릭터가 Mixamo로 명시, `datasets/__init__.py`에 하드코딩). SMPL 스켈레톤에 대해선 사전학습된 게 없어 **재학습 없이는 4-0에 못 씀**("학습 금지" 원칙과 충돌) — 하지만 이 저장소가 바로 **4-1(styled BVH→Mixamo)의 실제 후보**임을 확인(Mixamo가 이 도구의 원래 서식지).
- **참고 자료**: [CalciferZh/SMPL-AMC-Imitator](https://github.com/CalciferZh/SMPL-AMC-Imitator)(MIT)가 반대 방향(CMU AMC/ASF→SMPL 포즈)이지만 SMPL↔CMU 관절 대응표를 담고 있어 4-0b 스크립트 작성 시 참고 가능.
- **안심 근거**: Aberman/Li/Weng의 동일 저장소 자체가 "CMU mocap dataset의 표준 스켈레톤"을 스타일 전이 학습 데이터 포맷으로 쓰고 있다고 README에 명시 — Motion Puzzle이 요구하는 CMU-21 컨벤션이 이 연구 계보 전체가 공유하는 표준이라는 뜻. 즉 "임의로 까다로운 요구"가 아니라 이 바닥 공통 규격.

**결론**: 4-0은 **"변환기 찾기"가 아니라 "짧은 커스텀 리타겟 스크립트 작성"으로 확정** — 학습 없는 결정론적 기하 변환(SMPL 24관절 로컬 회전값 → CMU-21 해당 관절에 대응시켜 `skeleton_cmu.yaml` 고정 오프셋에 적용)이라 "조립" 원칙은 안 어김. 다만 이건 "도구 하나 clone하면 끝"이 아니라 **관절 대응표 + 리타겟 로직을 직접 짜야 하는 실제 개발 작업**이라는 걸 인지하고 세션 4 일정에 반영해야 함.

---

## 2. 세션별 파일 계획

| 세션 | 신규 파일(제안) | 재사용 | 데이터 계약 |
|---|---|---|---|
| 2-1 | `skeleton_extract.py` (신규 — `pose_extract.py` 확장 아님, 목적이 다름) | mediapipe 초기화 코드는 참고 가능 | 영상 → 33랜드마크 × 프레임 × (x,y,z) npz |
| 2-2 | — | `external/motion_puzzle/test.py` 그대로 호출 | BVH in/out |
| 2-3 | `analyze.py`에 함수 추가 또는 `evaluate_style_spike.py` 신규 | `dtw()` | styled BVH → 관절별 시계열 → DTW |
| 3-1/3-2 | Colab 노트북 `gvhmr_footmr_colab.ipynb` (신규) | — | 영상 → SMPL-X npz (footskate 보정 포함) |
| 4-0a | — | `smpl2bvh`(KosukeFukazawa, clone 필요) | SMPL-X body_pose → SMPL 24관절 회전(axis-angle) |
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
| smpl2bvh·deep-motion-editing용 env | 리타겟 2종(4-0a, 4-1) | **미생성이지만 Colab 불필요할 가능성 높음** — 둘 다 README·소스 확인상 Mac(CPU) 로컬 실행 가능해 보임(2026-07-23 확인). `motion_puzzle` env 재사용 가능한지(둘 다 PyTorch 기반) 다음에 확인 |
| Colab 노트북 1개 | **GVHMR + FootMR만** (같은 세션에서 이어서 실행) | **완료(2026-07-23)** — `footmr_colab.ipynb`, 실제 실행 성공. 겪은 문제 5가지(python3.10 부트스트랩, chumpy 빌드, tkinter 죽은 import, vitpose-h-wholebody 별도 링크, 파일명 특수문자)와 해결책 전부 노트북·`samsam_plan.md` 3절에 반영 |

---

## 4. 남은 결정 (2026-07-23 재확인 후 갱신)

- **해소됨**: SMPL-X→BVH(CMU) 경로 — "도구 확인 필요"였던 게 "smpl2bvh(4-0a, 실존 도구) + 커스텀 리타겟 스크립트(4-0b, 직접 작성)"로 확정. 4-1도 Aberman `deep-motion-editing`(Mixamo 사전학습 확인됨)으로 구체화.
- **해소됨**: pyenv vs conda — **conda, 이미 설치돼 있고 `motion_puzzle`/`mcmldm` env도 이미 존재**. 새로 설치할 것 없음.
- **해소됨(2절 반영 완료)**: 2-1(`skeleton_extract.py`)·6-2(`evaluate_icc.py`) 오늘 작성 + 자체검증 통과. 재사용 가능.
- **확인됨**: Motion Puzzle이 이 Mac에서 지금 당장 실제로 돌아간다(오늘 재실행 확인) — "로컬 실행 가능성 높음"이 아니라 사실로 확정.
- **해소됨(2026-07-23)**: 4-0b 커스텀 리타겟 스크립트 — 작성 완료, 합성 데이터로 Motion Puzzle 실제 파이프라인 통합까지 검증 통과. GVHMR 실측 데이터 나오면 바로 연결 가능한 상태.
- **해소됨(2026-07-23)**: 세션 5 three.js 뷰어 — `samsam_viewer.html` 프로토타입 작성 + Playwright 스크린샷으로 실제 렌더링 확인(BVH 직접 로드, 아직 Mixamo 리타겟 전).
- **여전히 미결**: smpl2bvh·deep-motion-editing 별도 env 필요 여부(각 repo clone 후 확인 필요) — 아직 clone 안 함.
- **여전히 미결**: Colab 노트북 착수 시점 — GVHMR·FootMR은 Google 계정 기반 Colab 실행이 필요해 이 세션에서 직접 실행은 못 함, 노트북 초안 작성은 가능.

---

*이 문서는 `samsam_development_plan.md`(전략)·`samsam_plan.md`(세션 로그)의 하위 문서 — 실제 파일·함수·포맷 단위 스펙만 담당.*
