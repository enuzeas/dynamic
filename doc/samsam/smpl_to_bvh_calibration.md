# 영상 → BVH 변환 보정값 (2026-09-11)

> 목적: `hmr4d_results.pt`(GVHMR/FootMR 출력)를 Motion Puzzle 입력 BVH로 바꿀 때 쓰는 보정 상수와 fps를 한곳에 적어둔다. 클립마다 다시 판단하지 않기 위한 문서다. `data/raw`에 171개 클립이 있고 앞으로 계속 변환해야 한다.

## 1. 왜 필요한가

첫 변환(`test_video`, 2026-09-11)에서 캐릭터가 바닥을 뚫고 상하로 크게 튀었다. 원인 두 가지였고 둘 다 "판단이 필요한 값"이라 여기 고정해둔다.

1. **단위** — CMU BVH는 cm가 아니라 자체 단위인데 `hmr4d_to_npz.py`가 미터→cm로 `×100`을 했다. 5.6배 과대.
2. **원점** — GVHMR의 `transl`은 절대 높이가 아니라 **첫 프레임 기준 상대 변위**다(실측 평균 0.06 m, 음수도 나옴). 바닥 높이를 더하지 않으면 캐릭터가 원점 근처에 떠 있다.

## 2. 보정 상수

`retarget_smpl_to_cmu.py`가 `RAW31_OFFSETS`에서 직접 계산한다. 하드코딩된 매직넘버가 아니라 골격에서 유도된 값이므로, 골격이 바뀌면 자동으로 따라간다.

| 상수 | 값 | 유도 |
|---|---|---|
| `SMPL_LEG_M` | 0.82 | SMPL 템플릿 다리 길이(고관절→무릎→발목), 미터 |
| `UNITS_PER_M` | **17.865** | (대퇴 \|offs[3]\|=7.158 + 경골 \|offs[4]\|=7.492) ÷ 0.82 |
| `REST_HIP_Y` | **15.993** | 힙→발목 Y 하강분 합 = `-RAW31_OFFSETS[2:6, 1].sum()` |

**검증 근거**: `REST_HIP_Y` 15.993은 CMU 참조 `test_bvh/41_02.bvh`의 루트 Y 최소값 **16.00**과 일치한다. 골격에서 유도한 값이 실제 CMU 데이터와 맞으므로 배율 근거가 선다.

적용 위치는 `retarget_smpl_to_cmu.py`의 `retarget()`:

```python
positions[:, 0] = root_trans * UNITS_PER_M
positions[:, 0, 1] += REST_HIP_Y
```

단위 변환과 바닥 정렬은 **골격을 아는 쪽**이 해야 하므로 여기에 둔다. `hmr4d_to_npz.py`는 SMPL 원래 단위인 미터를 그대로 내보낸다(예전엔 여기서 ×100을 했다 — 되돌리지 말 것).

## 3. fps — 원본별 실측값

`--fps`는 `hmr4d_to_npz.py`의 **필수 인자**다(`.pt`에 fps가 안 들어 있음). 원본 파일에서 확인한 값:

| 원본 | fps | 비고 |
|---|---|---|
| `data/raw/P0xx_*.mov` | **59.94** (`60000/1001`) | 1920×1080. 참가자 촬영본 전부 |
| 루트 `A_*.MOV` | 23.976 (`24000/1001`) | 3840×2160, 초기 테스트 |
| 루트 `B_*.MOV` | 24 (`24/1`) | 3840×2160, 초기 테스트 |

**함정**: `outputs/demo/<이름>/0_input_video.mp4`의 fps를 보고 판단하면 안 된다. `tools/demo.py:110`이 `get_writer(cfg.video_path, fps=30, crf=CRF)`로 **30을 하드코딩**한다. 원본 프레임을 전부 읽어 30fps로 라벨만 다시 붙여 쓰는 것이라 프레임 수는 보존되지만 fps 값은 원본과 무관하다. 반드시 `data/raw`의 원본을 `ffprobe`로 확인할 것:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 <원본>
```

fps가 2배 틀리면 동작 속도가 2배로 어긋난다 — 스타일의 타이밍 특성이 곧 우리가 잡으려는 신호이므로 치명적이다.

## 4. 변환 절차

```bash
conda run -n motion_puzzle python hmr4d_to_npz.py \
    --pt <hmr4d_results.pt> --out smpl_pose.npz --fps 59.94
conda run -n motion_puzzle python retarget_smpl_to_cmu.py \
    --npz smpl_pose.npz --out external/motion_puzzle/datasets/cmu/uploads/<이름>.bvh
```

`uploads/`에 넣으면 `style_transfer.html`의 두 드롭다운("내 업로드" 그룹)에 자동으로 뜬다. 파일명은 `^[A-Za-z0-9_.-]+\.bvh$`만 허용된다(한글·공백·괄호 불가).

Colab 출력은 구글 드라이브가 맥에 마운트돼 있어 따로 내려받을 필요가 없다:
`~/Library/CloudStorage/GoogleDrive-<계정>/내 드라이브/samsam_footmr_checkpoints/outputs/<이름>/hmr4d_results.pt`

## 5. 변환 후 검증

루트 Y가 CMU 참조 범위 안에 드는지 본다. 벗어나면 단위나 원점이 어긋난 것이다.

| 파일 | 루트 Y 최소~최대 | 수평 이동 X / Z |
|---|---|---|
| `127_21.bvh` (CMU 참조, 이동 많음) | 16.13 ~ 25.03 | 39 / 101 |
| `41_02.bvh` (CMU 참조, 제자리) | 16.00 ~ 16.82 | 32 / 34 |
| `test_video_20260911.bvh` (보정 전) | **−4.46 ~ 19.53** | **315 / 219** |
| `test_video_20260911.bvh` (보정 후) | 15.20 ~ 19.48 | 56 / 39 |

확인 스니펫:

```python
lines = open(path).read().splitlines()
i = next(k for k, l in enumerate(lines) if l.strip().startswith('Frame Time'))
print('Frame Time:', lines[i])                      # 59.94fps면 0.01668
rows = [l.split() for l in lines[i+1:] if l.strip()]
ys = [float(r[1]) for r in rows]                     # 루트 Y = 2번째 채널
print(f'루트 Y {min(ys):.2f} ~ {max(ys):.2f}')       # 대략 15~25 안이어야 정상
```

`Frame Time`이 0.01668(59.94fps)인지도 같이 본다. 0.03333이 나오면 `--fps`를 30으로 준 것이다.

## 6. 알려진 한계 (보정으로 해결 안 됨)

`retarget_smpl_to_cmu.py` 독스트링에 있는 두 가지가 그대로 남는다.

- SMPL spine2 회전을 버린다 — 등을 크게 굽히는 동작에서 오차.
- SMPL 쇄골(collar) 회전을 버린다 — 어깨를 크게 으쓱이는 동작에서 오차. CMU 골격에 쇄골 관절이 있지만 Motion Puzzle이 쓰는 21관절에 포함되지 않아, 채워도 결과에 반영되지 않는다.

스타일 레퍼런스용 클립을 고를 때 이 두 가지가 그 사람 스타일의 핵심인 동작은 피하는 편이 낫다.
