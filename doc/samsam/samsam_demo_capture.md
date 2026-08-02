# 쌤쌤 실전 촬영·처리 가이드 — 데모용 (2026-07-26)

> 목적: 다음 주 데모에서 **"우리가 직접 찍은 영상 → 스타일 전이"**를 보여주기 위한 촬영부터 파이프라인 통과까지 실전 가이드.
> 일반 촬영 원칙(조명·의상·동의 등)은 `samsam_shooting_guide.md`, 이 문서는 **GVHMR 파이프라인이 잘 먹히게 찍고 → BVH로 만들어 UI에 넣기**에 특화.

---

## 0. 큰 그림 — 영상 하나가 거치는 길

```
영상(.mp4)  →  GVHMR + FootMR      →  hmr4d_to_npz.py   →  retarget_smpl_to_cmu.py  →  CMU-BVH  →  스타일 전이 UI
(단안, 고정)   (Colab GPU, ~클립당 분)   (SMPL 24관절 npz)     (CMU 31관절 BVH)          (콘텐츠/스타일 입력)   (style_transfer.html)
```

앞단(GVHMR→BVH)은 전부 구현·검증돼 있음(`footmr_colab.ipynb`, `hmr4d_to_npz.py`, `retarget_smpl_to_cmu.py`). **영상→BVH는 Colab(GPU) 단계라 로컬 Mac에선 못 돎** — 촬영본을 Colab에서 BVH로 굽고, 그 BVH를 UI에 넣는다.

---

## 1. 촬영 요구사항 — GVHMR가 잘 먹히게 (기존 가이드 + 파이프라인 특화)

기본(고정 카메라·전신·밝은 조명·몸에 붙는 옷·단순 배경)은 `samsam_shooting_guide.md` 1절 그대로. 여기에 **GVHMR 특유의 요구**를 더한다:

| 항목 | 기준 | 왜 (GVHMR 내부) |
|---|---|---|
| 인원 | **프레임에 한 명만** | YOLO 사람 검출 — 여러 명이면 대상이 엉킴 |
| 프레이밍 | 전신이 매 프레임 안, 팔다리·발끝 안 잘리게 | ViTPose가 관절을 봐야 2D 키포인트가 나옴 |
| 카메라 | 삼각대 고정, 흔들림·줌 금지 | DPVO 카메라 추정 안정(고정이면 static_cam로 더 안정) |
| 조명 | 밝고 고름, 역광 금지 | 2D 키포인트 검출 실패 방지 |
| 발 | 신발·발끝 잘 보이게 | footskate(FootMR) 판정에 발 관절 필요 |
| 해상도/fps | 1080p+, 30~60fps. **fps를 반드시 기록** | 처리 뒤 `hmr4d_to_npz.py --fps`에 그대로 넣어야 타이밍이 맞음 |
| 길이 | **클립당 3~10초** | Colab 처리 시간·Motion Puzzle 윈도우에 적당(너무 길면 느리고 전이가 뭉개짐) |
| 파일명 | **영문·숫자·언더스코어만** (`walk_A.mp4`) | 공백·`()`·`@`는 Colab 셸/Hydra 파서가 깨짐(실측 확인된 문제) |

---

## 2. 무엇을 찍나 — 데모 임팩트 기준

스타일 전이 데모의 핵심은 **"같은 동작, 다른 결"**. 두 종류를 찍는다:

- **콘텐츠 클립 (무엇을 하는가)**: 걷기·뛰기·팔 휘두르기 등 **명확한 동작** 1~2개. 깔끔할수록 좋음.
- **스타일 클립 (어떤 결로)**: 결(자세·리듬·무게중심)이 확연히 드러나는 것:
  - **(추천) 팀원 2~3명이 같은 동작을 각자 방식대로** — "같은 걷기인데 사람마다 다르다"를 우리 데이터로 직접 시연. samsam 코어 메시지 그 자체.
  - 또는 **과장된 캐릭터 동작**(로봇처럼/닭처럼/터벅터벅) — 전이 효과가 극적으로 보임(공개 데이터에서 "닭 흉내 55_07", "공룡 137_11"이 잘 먹혔던 것처럼).
- **팁**: 첫 데모는 **"확실히 달라 보이는" 스타일 쌍을 의도적으로** 고를 것 — 2026-07-15 스파이크 교훈상 아무 쌍이나 잘 갈라지진 않는다(`samsam_plan.md` 2절).

파일명 규칙 예: `{이름}_{동작}.mp4` → `hanwoojin_walk.mp4`, `hanwoojin_robot.mp4`.

---

## 3. 영상 → BVH 처리 (클립 하나당)

1. **Colab**: `footmr_colab.ipynb` 열고 영상 업로드 → 셀 순서대로 실행 → `outputs/demo/{영상이름}/hmr4d_results.pt` 생성 → 다운로드.
   - (첫 실행은 python3.10·체크포인트 셋업으로 시간 걸림. 겪을 문제·해결책은 노트북·`samsam_plan.md` 3절에 전부 기록돼 있음.)
2. **로컬** (`conda activate motion_puzzle`), 프로젝트 루트에서:
   ```bash
   python hmr4d_to_npz.py --pt hmr4d_results.pt --out motion.npz --fps 60     # ← 촬영 fps 그대로
   python retarget_smpl_to_cmu.py --npz motion.npz --out motion.bvh
   ```
3. `motion.bvh`가 Motion Puzzle 입력(콘텐츠 또는 스타일) 포맷. 콘텐츠·스타일 클립 각각 이렇게 만든다.

---

## 4. UI에 넣기 — BVH 업로드 (구현됨, 2026-07-26)

`style_transfer.html`의 **＋ 업로드** 버튼으로 `motion.bvh`를 바로 올린다:
- 드롭다운 맨 위 "내 업로드" 그룹에 뜨고, 콘텐츠로 자동 선택됨(스타일로도 선택 가능).
- 백엔드가 CMU 형식(HIERARCHY·ROOT Hips·MOTION)만 통과시키고, 파일명은 영문·숫자·`._-`+`.bvh`만 허용(경로탈출 차단), 상한 50MB.
- 업로드본은 `external/motion_puzzle/datasets/cmu/uploads/`에 저장돼 다음 실행에도 남는다(gitignore 대상, 커밋 안 됨).
- (대안) 파일을 `.../test_bvh/`에 직접 복사하면 "전체(CMU ID)" 그룹에 떠서 코드 없이도 쓸 수 있다.

---

## 5. 이번 주 타임라인 (데모 D-7 기준)

| 시점 | 할 일 |
|---|---|
| D-6~5 | 촬영 — 팀원 2~3명, 콘텐츠+스타일 클립. 인당 15~20분 |
| D-4~3 | Colab 처리 — 클립별 GVHMR→BVH. **첫 클립을 일찍 끝까지 돌려 품질 확인**(단안 3D라 지터·발미끄러짐 가능) |
| D-2 | UI에 커스텀 BVH 넣고(업로드 또는 폴더 복사) 전이 결과 확인 → **좋은 콘텐츠·스타일 페어 선별** |
| D-1 | 폴백 준비(인터랙티브 UI 화면녹화), 오프라인·워밍업 리허설(`python style_transfer_server.py`) |

---

## 6. 동의·데이터 (건너뛰지 않기)

`samsam_shooting_guide.md` 5절 그대로: 용도(캡스톤 실험) 설명·동의, 철회 가능 고지, 움직임 데이터는 생체정보로 분류될 수 있어 검증 목적 외 재배포·공개 금지.

---

*참고: `samsam_shooting_guide.md`(일반 촬영 원칙), `samsam_dev_spec.md`(파이프라인·도구 스펙), `footmr_colab.ipynb`(GVHMR/FootMR 실행), `style_transfer.html`(스타일 전이 UI).*
