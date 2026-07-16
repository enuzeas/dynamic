# 파이프라인 코드 워크스루

영상 파일 → 관절별 다이내믹스 서명 → 검증 지표(rank-1/EER/ICC)/등록·인증(Dynamic ID)까지 코드가 실제로 하는 일을 순서대로 따라간다.

## 전체 흐름

```
영상(.mp4/.MOV)
  → extract_joint_dynamics()   [pose_extract.py]  MediaPipe pose landmark → 관절별 speed/accel/jerk 시계열
  → trim_motion()              [demo.py]           움직임 없는 앞뒤 구간 제거
  → 이후 용도에 따라:
     - plot_comparison()       [demo.py]              사람별 속도 곡선 나란히 비교
     - dtw + rank1/EER         [analyze.py]           고유성(식별 가능성) 검증 — 직접 촬영 표본
     - dtw (단일 관절)          [analyze_youtube_dance.py]  같은 챌린지 안무 유튜브 영상 다수 비교
     - enroll/verify/identify  [dynamic_id.py]        등록(Dynamic ID) → 1:1 검증 / 1:N 식별
     - icc_1_1                 [homeostasis_image.py] 반복 촬영 간 일관성(항상성) 검증
```

`demo.py`의 `extract_dynamics`는 `pose_extract.extract_joint_dynamics`의 별칭이다 — 모든 스크립트가 결국 이 한 함수로 관절 신호를 뽑는다.

## pose_extract.py — 관절 신호 추출 (신호의 원천)

**`extract_joint_dynamics(video_path, joints)`** (pose_extract.py:47)
MediaPipe `PoseLandmarker`(Tasks API, VIDEO 모드)로 프레임마다 33개 랜드마크를 검출하고, `LANDMARK_INDEX`에 정의된 오른손목(16)/오른팔꿈치(14)/오른어깨(12) 세 관절의 (x, y) 위치를 프레임 대각선 길이로 정규화해 저장한다(해상도가 다른 영상끼리 픽셀 이동량을 비교 가능하게 만들기 위함). 포즈 검출이 실패한 프레임은 직전 위치를 그대로 유지한다.

위치 시계열을 Savitzky-Golay 필터(`_smooth`, window=7, polyorder=3)로 평활화한 뒤 `np.gradient`로 미분해 속도를 얻고, 속도를 다시 한 번 평활화 후 미분해 가속도·저크를 만든다. 평활화 없이 위치를 두세 번 미분하면 포즈 추정 노이즈가 그대로 증폭되기 때문(코드 주석, pose_extract.py:4-5).

> **참고**: `_make_landmarker()`(pose_extract.py:30)는 `src/models/pose_landmarker_lite.task`를 읽는다. 이 파일은 `.gitignore`의 `*.task`에 걸려 저장소에 없고, `src/models/` 디렉터리 자체도 로컬에 존재하지 않는다 — 루트의 `pose_landmarker.task`(초기 커밋 자산, 다른 이름)와는 별개 파일이다. 실행 전에 해당 모델 파일을 받아 `src/models/` 아래 놓아야 한다.

## demo.py — 정렬·시각화

**`trim_motion(speed, threshold=0.08, pad=3)`** (demo.py:35)
속도가 threshold를 넘는 첫/마지막 프레임(+pad)만 남긴다. 촬영 시작 전/후의 정지 구간을 잘라내는 용도.

**`normalize_length(arrays)`** (demo.py:29) / **`plot_comparison(...)`** (demo.py:45)
길이가 다른 시계열을 최단 길이로 맞춰 평균 곡선을 만들고, 사람별 반복 촬영본 속도 곡선 + 평균±표준편차 비교 그래프를 그린다. CLI: `python demo.py --a A_1.mp4 A_2.mp4 --b B_1.mp4 B_2.mp4 --names 이름1 이름2`.

## analyze.py — 고유성(식별 가능성) 검증

**`dtw(s1, s2)`** (analyze.py:27) — 표준 1차원 DTW, O(n·m) DP.

**`load_samples` → `build_distance_matrix` → `rank1_accuracy`** (analyze.py:43/62/74)
사람×영상 표본 전체의 DTW 거리 행렬을 만들고, leave-one-out으로 "각 샘플의 최근접 이웃이 같은 사람인가"를 rank-1 정확도로 낸다.

**`sweep_far_frr(genuine, impostor, n=500)`** (analyze.py:89)
거리 임계값을 스윕해 FAR/FRR 배열을 반환하는 공용 함수. `compute_eer`(analyze.py:102)가 EER 계산에 쓰고, `dynamic_id.py`의 `calibrate_threshold`도 같은 함수를 그대로 재사용한다 — 임계값 로직이 한 곳에만 있다.

**`plot_results(...)`** (analyze.py:126) — DTW 거리 행렬, genuine/impostor 분포, FAR/FRR 곡선, 본인 내부(within) vs 타인 간(between) 거리 막대그래프 4패널.

CLI: `python analyze.py --people "이름1:A_1.MOV,A_2.MOV" "이름2:B_1.MOV,B_2.MOV"`

## dynamic_id.py — 등록(enroll)·검증(verify)·식별(identify)

`scope.md`가 말한 "BIO-IP 등록 Mock: 특징점 → 로컬 DB 저장·조회"를 실제로 동작하게 만든 것. 영상 원본이 아니라 관절별 속도 곡선 템플릿만 JSON으로 저장한다(로컬 `registry/` 디렉터리가 그 "DB").

- **`enroll(label, video_paths)`** (dynamic_id.py:55) — N개 등록 영상에서 관절별 속도 곡선을 뽑아 `DynamicID` 템플릿 생성 → `save_id`로 `registry/{label}.json` 저장
- **`verify(probe_video, id_, threshold)`** (dynamic_id.py:99) — probe와 등록 템플릿 중 최근접 반복까지의 DTW 거리(관절 평균)가 threshold 이하면 accept (1:1)
- **`identify(probe_video, registry)`** (dynamic_id.py:107) — registry 전체에서 가장 가까운 라벨을 찾음 (1:N)
- **`calibrate_threshold(registry)`** (dynamic_id.py:117) — 등록된 템플릿들 간 genuine/impostor 거리로 `sweep_far_frr` 기반 EER 임계값을 추정. 등록 인원·반복이 적으면(인당 2~3회) 추정이 거칠다는 `ponytail:` 주석 있음(dynamic_id.py:119) — `scope.md` 기준(8~12명, 인당 5회)까지 등록자가 늘면 재계산 필요.

CLI: `python dynamic_id.py enroll --label 이름 --videos a.mp4 b.mp4 --out registry/` / `verify --label 이름 --probe c.mp4 --registry registry/ --threshold 40` / `identify --probe c.mp4 --registry registry/` / `calibrate --registry registry/`

`test_dynamic_id.py`가 저장소 샘플(A_*, B_*)로 enroll→calibrate→verify→identify 전 과정을 돌리는 자체 검증 스크립트다.

## analyze_youtube_dance.py — 유튜브 안무 챌린지로 고유성 대규모 검증

`sample_dance/`(gitignore됨, 로컬 전용)에 있는 같은 챌린지 안무 영상 6개(크리에이터별 1개씩)를 오른손목 단일 관절 속도로 비교한다. 크리에이터당 영상이 1개뿐이라 반복이 없어 항상성 검증은 불가능하고, 서로 다른 사람 간 거리 분포(고유성)만 본다. `doc/plan.md`가 말하는 "안무로 콘텐츠가 통제된 자연 실험"을 실제 인터넷 데이터로 채운 사례.

## homeostasis_image.py — 항상성(반복 일관성) 검증

`pearsonr`/`icc_1_1`(homeostasis_image.py:26/33)로 같은 사람의 반복 촬영본 간 상관계수·ICC(1,1)를 계산한다. `PEOPLE` 리스트(homeostasis_image.py:19)에 파일 경로가 하드코딩돼 있어 CLI 인자 없이 코드를 직접 고쳐야 실행할 수 있다.

> **알아둘 것**: 이 스크립트는 `trim_motion`으로 자른 뒤 앞에서부터 최단 길이로 자르는 정렬만 쓴다 — `analyze.py`처럼 DTW로 정렬하지 않는다. `doc/plan.md` 3절이 바로 이 불일치("rank-1은 DTW 기반, ICC는 시작점 정렬 기반이라 정답률은 높은데 일관성은 낮다는 역설")를 지적하며 두 지표를 같은 정렬 기준으로 통일할 계획이라고 명시한다. 아직 코드에는 반영 안 됨.

## 현재 알려진 결과 (direction_brief.md 기준, MediaPipe 전환 이전 수치)

- Rank-1 100%, EER 13.9% (6샘플 소규모 실험)
- ICC 0.05~0.07 (항상성 기준선 0.5에 크게 못 미침)

이 수치는 pose_extract.py의 MediaPipe+평활화 전환 이전 것일 수 있다 — 전환 후 재측정 필요(`doc/next.md` 참고).
