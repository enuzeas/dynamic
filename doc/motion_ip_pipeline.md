# Motion IP 자산화 파이프라인 — 기획서 (검증된 기술 고도화 트랙)

> 위치: `scope.md`의 "Layer 3 — CG·애니·로봇 이식" 아이디어에서 출발하지만, Layer 1(고유성·항상성 MVP) 통과를 전제 조건으로 두지 않는다. 방향: 신규 연구를 설계하는 대신 **이미 검증되고 오픈소스로 존재하는 기술**을 그대로 가져와 이 프로젝트 데이터에 맞게 고도화한다. 새 이론을 증명할 필요가 없으니 Layer 1과 독립적인 병행 트랙으로 진행 가능 — 다만 최종적으로 "이 스타일 벡터가 개인 고유 자산"이라는 IP 주장의 근거는 여전히 Layer 1 지표(rank-1/EER/ICC)로 보강된다(5장).

---

## 1. 명칭에 대한 유의사항

아래 `SM-SGE` / `SA-PMT` / `PMSR`은 이 프로젝트에서 붙인 파이프라인 구성요소명이다. 새로 설계하지 않고, 각 구성요소는 이미 공개된 검증된 방법을 그대로 채택한다 — 아래 표가 "제안 명칭 → 실제로 가져다 쓸 것".

| 구성요소 | 채택할 검증된 기술 | 소스 |
|---|---|---|
| SM-SGE (스켈레톤 인코더 + 정제) | Motion Puzzle `Encoder_sty`(고정, 재학습 없음) 특징을 뽑아, 그 위에 Triplet Loss(Batch Hard, P명×K클립 배치) 정제 헤드를 얹음 | Jang et al. 2022 "Motion Puzzle"(이미 채택·실행 확인, `external/motion_puzzle`) + gait 논문(Zheng et al., arXiv:2111.11720)의 학습 방식 차용 — 처음 계획한 ST-GCN 처음부터 학습은 보류, 기존 특징 위 얕은 헤드로 우선 시도. **구현 후보(2026-08-01 조사)**: Zheng et al. 논문 자체는 코드 미공개(CASIA-B, ST-GCN, GitHub 확인 안 됨) → 직접 구현 대상 아님. Batch Hard+P×K 원조인 Hermans et al. 2017("In Defense of the Triplet Loss")의 참고 구현([CoinCheung](https://github.com/CoinCheung/triplet-reid-pytorch)/[kilianyp](https://github.com/kilianyp/triplet-reid-pytorch) triplet-reid-pytorch)은 손실 수식 확인용으로만 참고, 포크 대상 아님(오래됨·미관리). 실제 채택 후보는 **`pytorch-metric-learning`**(pip 설치, `miners.BatchHardMiner`+`losses.TripletMarginLoss`로 P×K 마이닝+로스 몇 줄에 해결, 유지보수됨) — `motion_puzzle`/`mcmldm` conda env 둘 다 미설치 확인, `pip install pytorch-metric-learning` 필요 |
| SA-PMT (스타일 전송) | **Motion Puzzle**(신체 부위별 AdaIN + attention) | Jang et al. 2022, ACM TOG — 실측 ~9초/쌍(`samsam_dev_spec.md` 89행). ~~Aberman et al. skeleton-aware retargeting~~은 리타겟팅 용도로만 검토했다가 이름 고정 24개 Mixamo 캐릭터 한정이라 폐기(`samsam_dev_spec.md`) — 스타일 전송 자체엔 애초에 쓴 적 없음 |
| PMSR (물리 정규화 손실) | Motion Puzzle이 이미 대부분 내장: 그래프 컨볼루션이 스켈레톤 토폴로지를 바꾸지 않아 `L_bone` 문제가 원천적으로 없고, `loss_sm_rec`/`loss_sm_cyc`(프레임 차분 L1)가 `L_jerk` 역할, `remove_fs.remove_foot_sliding()`이 `L_contact` 역할 — **별도 손실 함수를 새로 설계할 필요가 사라짐.** `L_limit`(관절 가동범위)만 미검증 잔여 리스크 | `external/motion_puzzle/trainer.py`(`compute_gen_loss`), `remove_fs.py` 코드 확인 |
| Sigma-Lognormal Model | kinematic theory of rapid human movements | Plamondon — 기존 오픈 구현체(SLM 툴박스) 재사용 |
| IKC | 표준 평균 궤적 대비 개인 편차 | 신규 지표 — 통계 계산이라 구현 난이도 낮음, 검증은 필요 |
| ICC | 급내상관계수 | `evaluate_icc.py::icc_3_1()`에 이미 구현됨(2026-07-25) — 현재 8관절 속도곡선(식별 트랙)에만 적용 중, SM-SGE 임베딩엔 배선만 하면 됨 |

심사·특허 문서에는 위 우측 컬럼(실제 논문·라이브러리명)으로 인용한다. 좌측은 이 프로젝트 내부에서 부르는 이름.

---

## 2. 왜 지금 착수 가능한가

새 알고리즘을 발명하는 게 아니라 검증된 오픈소스를 조립하는 작업이므로, Layer 1의 "다이내믹스 서명이 존재하는가"라는 명제 검증과 별개로 진행할 수 있다:

- **스켈레톤 인코더·리타겟팅·물리 손실** 모두 사전학습 가중치나 참조 구현이 공개돼 있어, 처음부터 학습 데이터를 모으지 않아도 파일럿이 가능하다 (사전학습 모델에 지금 가진 8–12명 × 5회 데이터를 얹어 fine-tune/평가하는 정도로 시작).
- 3D 스켈레톤 추정은 이 문서 작성 시점엔 "MediaPipe 3D world landmark로 교체" 계획이었으나, 실행 검증 후 **GVHMR+FootMR**(Colab T4)로 대체 확정됐다 — 정확도·world-grounded 지표에서 MediaPipe보다 우세(`samsam_plan.md` 76행). `hmr4d_to_npz.py`가 그 출력을 SM-SGE 입력용 npz로 변환하는 것까지 작성·검증 완료(2026-07-25).
- 다만 "스타일 벡터가 실제로 개인 고유 자산인가"라는 **IP 주장의 강도**는 Layer 1 지표로 뒷받침되는 게 맞다 — 파이프라인 구축 자체는 지금 해도 되지만, 특허/계약 문서에 "고유 자산"이라고 쓸 때는 rank-1/EER 숫자를 근거로 붙인다.

---

## 3. 파이프라인 아키텍처 (2026-08-01 실제 구현 기준으로 갱신)

```
[영상] → GVHMR+FootMR(2D→3D, Colab T4, 필수) → hmr4d_to_npz.py → retarget_smpl_to_cmu.py
       → [CMU BVH, K개 동작 클립 × 사람마다]
                │
                ▼
       Motion Puzzle Encoder_sty (고정, 재학습 없음)
                │  클립마다 5부위(다리L/R·척추·팔L/R) × 4스케일 특징
                ▼
       시간축 풀링 → 클립별 고정 벡터
                │
                ▼
       SM-SGE 정제 헤드 (신규 학습 — Triplet Loss, Batch Hard, P명×K클립)
                │
                ▼
       K개 임베딩 평균(centroid) = z_style (Motion IP 자산 후보)
                │                                    │
                │                      evaluate_icc.py::icc_3_1()로 반복촬영 간 일관성 검증
                ▼
       [타겟 모션 BVH] ──→ Motion Puzzle Decoder(BP-AdaIN+attention, SA-PMT) ──→ [스타일 이식된 모션]
                │
       remove_fs.remove_foot_sliding() + loss_sm_* (PMSR 역할, 아래 4장)
```

- **인코더(SM-SGE)**: Motion Puzzle의 `Encoder_sty`를 그대로 특징 추출기로 재사용(콘텐츠 인코더는 InstanceNorm으로 통계를 지우고 스타일 인코더는 안 지우는 구조로 이미 "동작 종류 대 통계적 스타일"을 어느 정도 분리함). 다만 이건 **클립 1개짜리** 통계일 뿐 "여러 다른 동작에 걸친 개인 불변 성분"은 아니라서, 그 위에 Triplet Loss 기반 정제 헤드를 새로 얹는 게 이 프로젝트의 실질적 신규 기여 지점(`professor_review_status.md` 4번 항목).
- **전송기(SA-PMT)**: Motion Puzzle `Decoder` — 신체 부위별 AdaIN(통계 이식) + attention(국소 패턴 이식)을 4단계 해상도에서 반복 적용. 원래 검토했던 Aberman skeleton-aware retargeting은 이 역할로 쓴 적 없음(리타겟팅 서브 문제에만 검토 후 폐기).
- **정규화(PMSR)**: Motion Puzzle 자체 손실(재구성+순환일관성+평활)과 `remove_fs`가 대부분 커버 — 아래 4장 참고.

---

## 4. PMSR 손실 함수 정의

**갱신(2026-08-01)**: Motion Puzzle을 그대로 채택하면서 아래 손실 대부분을 새로 짤 필요가 없어졌다 — `L_bone`은 스켈레톤 토폴로지를 안 바꾸는 그래프 컨볼루션 구조상 원천적으로 문제가 안 생기고, `L_jerk`는 `loss_sm_rec`/`loss_sm_cyc`, `L_contact`는 `remove_fs.remove_foot_sliding()`이 이미 대신한다(1장 표 참고). 아래 정의는 **`L_limit`처럼 아직 안 커버되는 항이 실제로 필요해질 경우를 위한 참고 정의**로 남겨둔다 — 지금 당장 구현 우선순위는 아니다.

물리적 개연성을 강제하는 항의 가중합으로 정의한다. 개별 항은 리타겟팅/모션 합성 연구에서 흔히 쓰는 정규화를 이 프로젝트 맥락에 맞춘 것이다.

```
L_total = L_task + λ_bone · L_bone + λ_jerk · L_jerk + λ_contact · L_contact + λ_limit · L_limit
```

| 항 | 정의 | 목적 |
|---|---|---|
| `L_bone` | 본 길이(관절 간 거리)의 프레임 간 분산 | 골격 안정성 — 팔다리가 늘었다 줄었다 하는 왜곡 방지 |
| `L_jerk` | 관절 가속도의 시간 미분(저크) 크기에 대한 패널티 | 최소 저크 이론(`BIO-IP_통합문서.md` 4장) 근거 — 부자연스러운 급변 억제, 동시에 다이내믹스 서명 자체가 저크 기반이므로 과도한 평활화로 서명을 지우지 않게 λ 조절 필요 |
| `L_contact` | 지면 접촉이 필요한 관절(발 등)의 위치 일관성 | 미끄러짐(footskate) 방지 |
| `L_limit` | 관절 각도가 인체 가동 범위를 벗어나는 정도 | 신체 연결성·해부학적 타당성 |
| `L_task` | 목표 동작(콘텐츠)과의 유사도 (예: 관절 위치 재구성 오차) | 원래 동작 의미 보존 |

**학습 안정성 확보 논리**: `L_task`만으로 학습하면 스타일 벡터가 극단적으로 반영되면서 골격이 붕괴하는 방향으로 학습이 발산하기 쉽다. `L_bone`·`L_limit`은 매 스텝마다 "물리적으로 가능한 해"로 그래디언트를 되돌리는 역할을 하므로, 학습 초반부터 `λ_bone`, `λ_limit`을 상대적으로 높게 주고(warm-up 이후 서서히 낮춤) 스타일 반영 가중치를 키우는 스케줄링이 필요하다. `λ_jerk`는 반대로 너무 크면 다이내믹스 서명(개인성)을 뭉갤 수 있어 가장 보수적으로 조정해야 한다 — 이 지점이 "재현이 서명을 보존하는가"라는 검증 질문과 직결된다.

---

## 5. Motion IP 자산 정의와 계약 요소

- **뇌-신경 운동 제어 프로필**: SM-SGE의 스타일 벡터 `z_style` 자체. 이 벡터의 클러스터 내 분산(반복 촬영 간 일관성)이 신경 변동성 지표 후보 — `analyze.py`가 이미 계산하는 within-subject 거리와 개념적으로 동일한 검증이 필요.
- **IKC**: 표준 평균 궤적(다수 피험자 평균) 대비 개인 궤적의 편차. 정의는 `(개인 관절 궤적) − (집단 평균 궤적)`의 시계열 노름. 신규 지표이므로 계산식·정규화 방법을 별도로 검증해야 하며, 현재 프로젝트의 "between-subject 분류 정확도(고유성)"와 다른 각도의 지표다 — 둘 다 필요할 수 있으나 중복 여부 확인 필요.
- **ICC**: 이미 구현된 지표 재사용. 기준값 0.85는 임상 신뢰도 연구에서 흔히 쓰는 "우수(excellent)" 경계값과 일치 — 도용 여부 판정 임계값으로 쓸 근거는 있으나, BIO-IP MVP의 항상성 검증(rank-1/EER 대신 ICC)과 역할이 겹치지 않는지 정리 필요.

---

## 6. 선행 조건 및 리스크

1. 3D 스켈레톤 추정 정확도 리스크는 GVHMR+FootMR 채택으로 어느 정도 줄었다(MediaPipe 3D landmark보다 world-grounded 지표 우세). 다만 잔여 리스크가 남는다: Motion Puzzle의 학습 데이터(Xia/CMU/edin_locomotion)가 걷기·뛰기·점프 같은 로코모션 위주라, 촬영 계획(`samsam_shooting_guide.md`)에 있는 손짓·인사·조는 척 같은 동작에서는 스타일-콘텐츠 분리 자체가 학습 분포 밖이라 품질이 떨어질 수 있다 — SM-SGE 정제 헤드 설계·검증 시 이 도메인 격차를 전제로 둬야 한다.
2. `BIO-IP_통합문서.md` 5장의 "재현 위협"과 동일한 문제: 이 파이프라인이 완성되면 곧 "서명을 훔쳐 다른 사람인 척 재현"하는 것도 기술적으로 가능해진다. 권리 인프라(동의·워터마킹) 설계가 파이프라인 구현보다 먼저 또는 동시에 필요하다.
3. 이 문서의 acronym들을 대외 발표 자료에 쓸 경우 실제 인용 가능한 논문·라이브러리명(1장 표 우측)으로 대체할 것.
4. "파이프라인이 돌아간다"와 "스타일 벡터가 개인 고유 자산이다"는 다른 주장이다 — 후자는 Layer 1 지표 없이는 방어할 수 없으므로, IP 계약 문서 작성 시점에는 Layer 1 데이터가 필요하다.

---

## 7. 다음 단계 (2026-08-01 갱신 — 1~3번 항목은 완료·대체돼 아래로 교체)

~~기존 1~3번(MediaPipe 3D landmark 교체, ST-GCN 사전학습 파일럿, Aberman 리타겟팅 검증)은 각각 GVHMR 채택, Motion Puzzle Encoder_sty 재사용 결정, Aberman 리타겟팅 폐기로 완료·대체됨.~~ 남은 실질 작업:

1. Motion Puzzle `Encoder_sty` 특징을 시간축 풀링해 클립별 고정 벡터로 만드는 전처리 스크립트 작성(신규, 작은 공수).
2. 그 위에 정제 헤드(MLP + Triplet Loss, Batch Hard, P명×K클립 배치) 프로토타입 학습 — 데이터는 `samsam_shooting_guide.md`의 8~12명×8동작 계획 그대로 사용.
3. 학습된 임베딩에 `evaluate_icc.py::icc_3_1()` 연결해 반복 촬영 간 일관성 확인 — 도구는 이미 있음, 배선만 남음.
4. ICC가 낮게 나오면: (a) 정제 헤드를 얕은 MLP에서 ST-GCN 처음부터 학습으로 확장하거나, (b) 대규모 공개 데이터(많은 사람×많은 동작)로 먼저 사전학습 후 우리 데이터로 파인튜닝 — 어느 쪽이 필요한지는 3번 결과를 보고 판단. (b) 후보(2026-08-01 조사): **NTU RGB+D 60**(40명×60동작) / **120**(106명×120동작, subject 수 많아 P×K에 적합) — 3D 관절 좌표라 Motion Puzzle 입력(BVH 회전)과 좌표계 다름, 변환 필요. 대안으로 **BABEL**(AMASS 기반, subject 라벨 있음, mocap이라 BVH/CMU 골격 호환성이 NTU보다 나을 가능성 — 검토 안 해본 옵션)도 후보. PKU-MMD·Human3.6M은 규모/인원 부족으로 우선순위 낮음.
