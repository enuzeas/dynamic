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
   └─ (4-1) ─────────────────────────────→ styled BVH → 회전 리타겟(이름매핑 + A_parent⁻¹·Q·A_i + T-pose 정렬) → X Bot(three.js 뷰어)
```

### 4-0a 최종 결정 (2026-07-25, smpl2bvh 실제 clone·실행 후 뒤집음)

**원래 계획(2026-07-23, 아래 "폐기된 경로" 절 참고)은 smpl2bvh로 "SMPL 24관절 BVH"를 만드는 것이었다. 실제로 clone해서 돌려본 결과 이 단계 자체가 불필요하다는 게 드러나 계획을 바꿨다:**

1. **FootMR 소스를 직접 읽어 GVHMR 출력의 실제 스키마를 확인함**(`external/FootMR/hmr4d/model/footmr/pipeline/footmr_pipeline.py`, `footmr_pl_demo.py`) — `hmr4d_results.pt`의 `smpl_params_global` 딕셔너리는 `global_orient`(F,3)·`body_pose`(F,63 = 21관절 축각)·`transl`(F,3)이고, 이 21관절 순서가 `retarget_smpl_to_cmu.py`의 `SMPL_JOINT_NAMES[1:22]`와 정확히 같은 표준 SMPL 순서다. 즉 GVHMR/FootMR 출력은 **이미 로컬 축각 회전**이라 forward kinematics도 SMPL 본체 파일(라이선스 걸림)도 필요 없이 그대로 이어붙이면 4-0b 입력이 완성된다.
2. **smpl2bvh를 실제로 clone·실행해 검증하는 과정에서 그쪽 자체의 버그를 발견함** — `mcmldm` conda env(smplx 이미 설치됨)에서 돌리려면 chumpy가 numpy≥1.24에서 제거된 `np.bool`/`np.int` 등을 참조해 깨지는 것부터 몽키패치로 우회해야 했고, 우회 후 합성 데이터(오른쪽 팔꿈치만 굽힌 60프레임)로 실제 BVH를 뽑아보니 **회전이 정확히 0인 관절마다 NaN이 나옴**(`utils/quat.py`의 `from_axis_angle()`이 `axis = rots / angle`에서 0/0 나눗셈 — 정지 관절이 하나라도 있으면, 즉 보통 모션 대부분에서 발생). `retarget_smpl_to_cmu.py`가 쓰는 `Quaternions.from_angle_axis`(motion_puzzle 자산)는 이미 0벡터를 안전 처리해서 이 문제가 없다.

**결론**: smpl2bvh clone은 `external/smpl2bvh`에 참고·시각 디버그용으로만 남겨두고(파이프라인 제외), 4-0a는 **`hmr4d_to_npz.py`**(신규, 20줄 안팎의 순수 배열 변환 — `global_orient`+`body_pose`를 이어붙이고 손(SMPL 22·23, GVHMR 출력에 아예 없음)은 항등회전, `transl`은 ×100해서 CMU BVH의 cm 스케일에 맞춤)로 확정. SMPL 모델 파일도, smplx 패키지도, 별도 env도 필요 없어져 "여전히 미결"이었던 4-0a용 env 문제 자체가 사라짐.

### 4-1 최종 해법 (2026-07-26, 여러 번 뒤집은 끝에 확정 — 전 관절 dot=1.0 검증)

styled CMU-BVH의 회전을 Mixamo 휴머노이드에 입히는 리타겟. **사전학습 딥러닝 리타겟(deep-motion-editing)은 폐기**(맨 아래 근거), 대신 **결정론적 회전 리타겟**으로 확정했다. 여기까지 오는 데 함정이 많아 전부 기록한다 — 최종 해법은 아래 3가지의 조합이다.

**(1) 깨끗한 리그로 교체 — Soldier.glb → Mixamo X Bot(FBX).**
처음엔 three.js 예제의 `Soldier.glb`를 썼는데, 이 리그가 좌표프레임이 엉망이라 어떤 리타겟 방법도 안 맞았다(실측으로 확인한 3중고): ① 조상 노드 `Z_UP`이 -90° X회전을 걸어 본들이 Y-up이 아닌 프레임에 살고, ② 아마추어에 100배/0.01배 중첩 스케일이 박혀 있고, ③ 힙 rest에 180° Y회전이 베이크돼 있었다. 이 셋이 겹쳐 카메라·회전·스케일 버그를 줄줄이 만들었다. **Mixamo 기본 캐릭터 X Bot을 FBX(T-pose)로 받아 교체하니 리그가 깨끗했다** — 실측: Y-up, 힙 worldScale=1, 힙 rest 회전 ≈ 항등, 오른팔 rest 방향 -X(= CMU BVH 오른팔 방향과 일치). Mixamo는 glTF를 안 주므로 FBX로 받아 `FBXLoader`로 직접 로드(변환 도구가 좌표프레임을 또 건드리는 걸 회피). `viewer_data/xbot.fbx`.

**(2) 올바른 회전 공식 — ℓ_i = A_parent(i)⁻¹ · Q_src_i · A_i.**
`A_i` = 타겟 본 i의 rest **월드** 누적 회전(= `bone.getWorldQuaternion()` at rest), `Q_src_i` = CMU 본 i의 로컬 회전. CMU는 rest가 전부 항등이라 rest 월드도 항등 → 소스의 "rest 대비 델타"가 곧 절대 월드회전이 되고, 그걸 타겟 rest 월드 위에 얹어 로컬로 되돌리면 이 식. **처음에 쓴 `ℓ = R_rest_local · Q`(부모 누적회전 무시)가 틀린 공식이었고, 이게 "다리 꼬임·팔 휘적"의 진짜 원인**이었다(팔·다리처럼 부모가 항등이 아닌 사슬에서 전부 어긋남). 올바른 공식으로 바꾸니 **팔은 dot=1.0(완벽)**, 다리는 dot≈0.94로 개선.

**(3) T-pose 방향 정렬 — 남은 다리 dot 0.94를 1.0으로.**
팔은 우연히 CMU와 X Bot의 rest 팔 방향이 일치(둘 다 -X)해 (2)만으로 완벽했지만, **X Bot rest 다리는 CMU보다 ~20° 벌어져(A-스탠스)** 있어 dot 0.94가 남았고, 그 20° 차이가 무릎 굽힘 평면을 돌려 깊은 스쿼트에서 다리가 겹쳤다. 방향 dot로는 이 twist가 안 잡혀서 처음엔 놓쳤다. 해법: 각 사지 본에 대해 **rest 월드 방향(부모→자식)을 소스에 맞추는 최소회전** `align_i = Quaternion.setFromUnitVectors(타겟rest방향, 소스rest방향)`를 구해 `M_i = align_i · A_i`로 교체(팔은 align≈항등이라 그대로 완벽 유지). 정렬 대상은 사지 10본(UpLeg·Leg·Foot·Arm·ForeArm 좌우). **결과: 허벅지·정강이·팔 전부 dot=1.000, 전 프레임** — 다리 교차 소멸. 매핑 표·align·검증은 `samsam_viewer.html`.

**(4) 힙 이동(점프·전진) 추가 (2026-07-26).**
회전만 넣었을 땐 제자리 재생이라 소스의 점프·전진이 안 보였다("모션퍼즐처럼 점프를 안 한다"는 피드백). 힙 이동 트랙을 추가: `hipLocal(t) = restLocal + (bvhHip(t) - bvhHip(0)) · scale`. X Bot·CMU 둘 다 Y-up·같은 월드 방향(회전 dot=1.0로 확인)이라 **축 스왑 없이 성분 그대로** 매핑(Soldier 때의 x,z,y 스왑·부호 뒤집기가 전부 불필요). `scale = X Bot 다리길이(cm) / CMU 다리길이(BVH단위)`로 점프높이·보폭이 몸 크기에 비례. 이 클립은 전진이 ~6.8m로 커서(실측: 소스 힙 Z 0.2→109단위) **카메라가 힙의 수평 이동을 매 프레임 따라가게** 함(캐릭터는 프레임에 크게 유지, 그리드가 흘러 전진이 보이고, 점프는 프레임 안에서 수직으로 보임; target·카메라를 같은 델타로 옮겨 OrbitControls 드래그 회전은 유지).

**현재 상태**: 회전 정확(전 관절 dot=1.0) + 점프·전진 재현 완료. 발 고정(footskate 보정)은 남은 과제(지금도 착지·서기는 자연스러움).

### 겪은 함정 (같은 삽질 반복 방지용 기록)

- **본 이름 콜론 제거**: `FBXLoader`·`GLTFLoader` 둘 다 로드 시 본 이름의 콜론을 뗀다 — 원본 `mixamorig:Hips` → 로드 후 `mixamorigHips`. 콜론 버전으로 매핑 표를 짰다가 "No target node found" 경고로 전 관절이 안 움직였다.
- **Neck vs Neck1**: Motion Puzzle 출력 BVH(21관절 축소판)에는 `Neck`이 아니라 `Neck1`만 있다(+`LowerBack`·`Shoulder` 좌우·손가락 상세 없음). `Neck`으로 매핑해 목이 내내 안 움직였다.
- **패널별 Clock 공유 금지**: 4개 패널이 `THREE.Clock` 하나를 공유하면 같은 rAF 프레임 안에서 나중에 `getDelta()`하는 패널일수록 델타가 0에 수렴해 멈춘 것처럼 보인다. 시계 하나(`masterClock`)로 통합하고 각 패널은 로드 완료 시점의 경과시간을 따라잡게 함.
- **retargetClip(three.js SkeletonUtils) 안 씀**: 표준 도구라 시도했으나 (a) Soldier에선 `bone.matrix.decompose()`가 본 스케일을 건드려 놓고 복원 안 해 메쉬가 폭발(중첩 스케일 때문, `skeleton.pose()`로 우회 가능했지만), (b) 깨끗한 X Bot에서도 bare Skeleton을 소스로 넘기면 baking이 깨져 다리 dot=-0.94(반대)·팔 dot=0.00(수직)로 오히려 더 나빴다. 손수 짠 (2)+(3) 공식이 전 관절 1.0으로 압승.

### 폐기된 후보 (기록용)

- **[DeepMotionEditing/deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing)** (Aberman et al., BSD-2-Clause, clone 완료) — 사전학습 리타겟 네트워크는 학습·평가 때 **이름이 고정된 24개 Mixamo 캐릭터**만 다룬다(`retargeting/datasets/__init__.py`의 `get_character_names()`). `combined_motion.py`가 캐릭터별 정규화 통계(`mean_var/{character}_mean.npy`)·기준자세(`std_bvhs/{character}.bvh`, 둘 다 Google Drive 데이터셋 안)를 읽어, 우리 CMU 골격 같은 새 스켈레톤은 재학습 없이 못 씀("학습 금지" 원칙 위반) → 폐기.
- **`Soldier.glb`** — 위 (1)의 3중고로 폐기, `viewer_data/`에서 삭제함(2026-07-26). 필요하면 three.js 예제 저장소에서 다시 받을 수 있음.
- **[KosukeFukazawa/smpl2bvh](https://github.com/KosukeFukazawa/smpl2bvh)** — 4-0a 후보였다 폐기(위 "4-0a 최종 결정" 절), 4-1과 무관.

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
| 4-1 | **`samsam_viewer.html`에 통합 — 회전+힙이동 완료·검증(2026-07-26)** | `viewer_data/xbot.fbx`(Mixamo X Bot, `FBXLoader`) | CMU BVH 회전 → 이름 매핑 + `ℓ=A_parent⁻¹·Q·A_i` + T-pose 방향 정렬(align) → mixamorig 구동, **전 관절 dot=1.0**. 힙 이동(점프·전진) 트랙도 추가(축 스왑 없이 성분 매핑, 카메라 수평 추적). 발 고정만 남음 |
| 5 | `samsam_viewer.html` (신규, 프로젝트 폴더 아니라 단일 파일 — 빌드 없는 정적 사이트 컨벤션 유지) | 없음 — `reports/viewer.html`은 마크다운 리포트 뷰어, `index.html`/`mvp.html`은 2D MVP라 3D 스켈레톤과 무관 | **2026-07-26 데모 뷰어로 완성**: 3패널 모두 X Bot 캐릭터(스켈레톤 아님) — 원본 콘텐츠(127_21)/스타일 A/스타일 B를 같은 캐릭터에 리타겟해 "같은 동작, 다른 스타일"을 before/after로 비교. `SkeletonUtils.clone`으로 캐릭터 3개 복제, 리타겟 rig(km+align+hip)는 `computeRig()`로 한 번 계산해 공유, `masterClock`로 동기화, 패널별 힙 수평 추적 카메라 |
| (툴) | **`style_transfer_server.py` + `style_transfer.html` — 인터랙티브 스타일 전이 UI(2026-07-26)** | `test.py`(subprocess), 4-1 리타겟 코드(HTML에 복사) | 콘텐츠 BVH + 스타일 BVH 선택 → 백엔드가 `conda run -n motion_puzzle python test.py`로 Motion Puzzle 실행(~9초) → 결과 BVH를 X Bot 캐릭터로 표시. 파이토치·conda가 필요해 브라우저 단독 불가라 로컬 백엔드(stdlib http.server, 포트 8940). 입력 풀은 `datasets/cmu/test_bvh/*.bvh`(270개). 드롭다운은 2단 — "추천 동작(이름)" 12개(걷기·뛰기·점프·발차기·복싱·춤·닭 흉내·공룡 등, CMU 공식 설명을 [una-dinosauria/cmu-mocap 인덱스](https://github.com/una-dinosauria/cmu-mocap)에서 받아 한국어 라벨) + "전체(CMU ID)" 270개. 기본 데모 페어=걷기+닭 흉내. 영상→BVH는 GVHMR(Colab) 단계라 UI 범위 밖(명시). `python style_transfer_server.py`로 실행 |
| 6-2 | `evaluate_icc.py` (신규 — ICC 구현이 코드베이스에 전혀 없음, 확인함) | `analyze.py`의 DTW | styled BVH 여러 개 → ICC 계수 |

---

## 3. 환경 구성

| 환경 | 용도 | 상태 |
|---|---|---|
| 로컬 `.venv` (Python 3.14, 기존 `requirements.txt`) | mediapipe 포즈 추출, DTW/ICC 평가 | 이미 있음, 그대로 사용. `skeleton_extract.py`·`evaluate_icc.py` 신규 작성 + 자체검증 통과(2026-07-23) |
| conda env `motion_puzzle` (Python 3.8.20, PyTorch 2.4.1) | 스타일 전이 추론 | **이미 존재 + 오늘 실제로 재실행해서 확인함.** `.cuda()` 하드콜 없음(이미 `torch.device(... if is_available ...)`로 패치돼 있음), CMU 샘플로 `test.py` 정상 실행·BVH 출력 확인(`external/motion_puzzle/output/dev_verify_test/`) — "pyenv vs conda 결정 필요"는 해소됨(conda, 이미 있는 걸 그대로 씀) |
| conda env `mcmldm` (Python 3.9.23) | MCM-LDM(2순위 후보) | 이미 존재, 오늘은 미실행(코어 아님) |
| (해당 없음) | 리타겟(4-1) | **불필요(2026-07-26)** — 리타겟이 브라우저 JS(three.js `FBXLoader` + 회전 공식)로 끝나 별도 conda env·파이썬 도구 자체가 필요 없음. 입력 캐릭터는 `viewer_data/xbot.fbx`(Mixamo X Bot, T-pose FBX) |
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
- **해소됨(2026-07-26 최종)**: 4-1 회전 리타겟 — Soldier.glb 리그 문제로 여러 번 뒤집은 끝에 "Mixamo X Bot(FBX) + `ℓ=A_parent⁻¹·Q·A_i` + T-pose 방향 정렬"로 확정. 전 관절 방향 dot=1.0 검증. 근거·함정은 위 "4-1 최종 해법" 절.
- **해소됨(2026-07-26)**: 4-1 힙 이동 — 점프·전진 트랙 추가 완료(축 스왑 없이 성분 매핑, 카메라 수평 추적). 발 고정(footskate)만 남음.
- **여전히 미결**: Colab 노트북 착수 시점 — GVHMR·FootMR은 Google 계정 기반 Colab 실행이 필요해 이 세션에서 직접 실행은 못 함, 노트북 초안 작성은 가능.

---

*이 문서는 `samsam_development_plan.md`(전략)·`samsam_plan.md`(세션 로그)의 하위 문서 — 실제 파일·함수·포맷 단위 스펙만 담당.*
