# professor_review.md 주요 세부기술 — 실제 구현 현황 (2026-08-01)

> 목적 한 줄: `professor_review.md`의 "주요 세부기술" 1~9번이 실제 코드/실행 기준으로 **해소됐는지, 도구는 있는데 연결만 안 됐는지, 완전히 비어있는지**를 정리한다. 교수 면담 준비용 — 리뷰 문서의 청사진과 지금 코드베이스 사이의 격차를 짚는 문서.

---

## 1. 항목별 현황

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | 영상 기반 아바타 생성(외형 재현) | **미해소** | 코드 전체에 아바타 생성 파이프라인 없음. 뷰어는 고정 Mixamo X Bot 캐릭터만 사용 |
| 2 | 2D/3D 모션 캡처(MediaPipe+SMPL-X) | **해소 — 문서보다 나은 방법으로 대체** | raw MediaPipe 3D landmark 대신 **GVHMR+FootMR** 채택(`samsam_plan.md` 76행 — 정확도·world-grounded 지표 우세, 핵심망 추론 WHAM 대비 7배 빠름). `hmr4d_to_npz.py`가 그 출력을 SMPL 24관절 축각으로 변환 완료(2026-07-25) |
| 3 | 신체 부위별 스타일 인코딩·전이(Motion Puzzle) | **해소** | `external/motion_puzzle` 그대로 채택, 실측 ~9초/쌍(`samsam_dev_spec.md` 89행). `Encoder_sty`+`Decoder`(body-part 그래프, `model.py`) 코드 확인 |
| 4 | 개인 고유 특징 정제(동작-종속 변이 분리 + 다중 샘플 집계) | **설계 완료(2026-08-01), 구현 후보 확정, 구현 전** | Motion Puzzle 자체엔 이 기능 없음(입력 BVH 1개 → 스타일 코드 1개, 다중 샘플 집계 로직 부재). 대신 gait 논문(Zheng et al., arXiv:2111.11720)의 Triplet Loss(Batch Hard, P명×K클립) 방식을 Motion Puzzle `Encoder_sty` 고정 특징 위에 얹는 구조로 설계 확정 — 상세는 `motion_ip_pipeline.md` 3·7장. 학습 코드 후보 조사 완료(2026-08-01): Zheng et al. 원 논문 코드 미공개, 원조 격인 Hermans et al. 2017 구현체는 참고용, 실제 채택은 **`pytorch-metric-learning`**(pip, 미설치 확인)로 확정 — 상세는 `motion_ip_pipeline.md` 1장. 아직 코드로 옮기진 않음. Motion Puzzle 학습 데이터(Xia/CMU/edin_locomotion)가 걷기·뛰기·점프 위주라 손짓·인사·조는 척 동작엔 도메인 격차 잔존 |
| 5 | 일관성 검증(ICC) | **부분 해소 — 도구는 있음, 배선 대상 확정** | `evaluate_icc.py`에 `icc_3_1(data)` 범용 함수 구현 완료(2026-07-25). 4번 설계에서 "정제 헤드가 뽑은 K개 임베딩의 centroid"에 연결하는 지점까지 확정(`motion_ip_pipeline.md` 3장 다이어그램) — 실제 배선(코드 연결)만 남음 |
| 6 | 개인 IP 임베딩 스토어 | **미해소** | 저장·식별자·버전관리 코드 없음. 매번 BVH 파일 쌍으로 즉석 실행(`style_transfer_server.py`) |
| 7 | 아바타-스타일 결합 및 전이 | **모션 쪽 리타겟은 해소, "결합" 자체는 검증 대상이 없음** | `retarget_smpl_to_cmu.py`(4-0b, 완료)·CMU→Mixamo 회전 리타겟(4-1, 완료 — 전 관절 방향 dot=1.0 검증, 발 고정만 남음, `samsam_dev_spec.md` 87행)까지는 됨. 다만 이건 "모션 스켈레톤→모션 스켈레톤" 리타겟이지, 문서가 우려하는 "독립적으로 생성된 아바타 메시와 모션의 결합"은 1번(아바타 생성)이 없어서 **테스트할 대상 자체가 아직 없음** |
| 8 | 재현 신뢰도 검증(DTW) | **부분 해소 — 도구는 있음, 배선만 안 됨** | `src/analyze.py`에 `dtw(s1, s2)` 범용 함수로 이미 구현, 다른 트랙(`takemeup_dance_analysis.md` 등)에서 실사용 중. Motion Puzzle 출력(원본 vs 재현)에는 아직 미적용 |
| 9 | 실시간 렌더링/뷰어(three.js) | **해소** | `samsam_viewer.html`/`style_transfer.html` 작동 확인. 인터랙티브 UI(`style_transfer_server.py`, 포트 8940)까지 완료 |

---

## 2. 요약

9개 중 **완전 해소 3개**(2, 3, 9), **설계 완료·구현 전 1개**(4), **도구는 있고 연결만 남은 것 2개**(5, 8), **절반만 해소 1개**(7 — 모션 쪽만), **완전 미해소 2개**(1, 6).

`professor_review.md`가 "핵심 병목·독창적 기여 지점"이라 부른 **4번(개인 고유 특징 정제)**은 미해소에서 설계 완료 단계로 진전됨(2026-08-01) — Motion Puzzle 원 논문에 없는 기능이라 직접 설계했고, 구조는 `motion_ip_pipeline.md`에 반영됨. 다음 단계는 이 설계를 코드로 옮기는 것.

---

## 3. 남은 일 우선순위 (제안, 2026-08-01 갱신)

1. **4번 구현 착수** — 설계는 끝남(`motion_ip_pipeline.md` 7장): Motion Puzzle `Encoder_sty` 특징 풀링 → `pytorch-metric-learning`(`BatchHardMiner`+`TripletMarginLoss`)로 정제 헤드 학습 → centroid 집계. 남은 건 코드로 옮기는 것
2. **5·8번 배선** — 이미 있는 `icc_3_1`·`dtw` 함수를 4번 결과(정제된 임베딩)와 Motion Puzzle 출력에 연결(구현 자체는 재사용, 공수 작음) — 4번 구현과 사실상 같은 작업 묶음
3. **4번 검증 후 분기** — ICC가 낮게 나오면 (a) 정제 헤드를 ST-GCN 처음부터 학습으로 확장하거나 (b) NTU RGB+D(60/120) 또는 BABEL 같은 대규모 데이터로 사전학습 후 파인튜닝(후보 비교는 `motion_ip_pipeline.md` 7장)
4. **1번(아바타 생성)** — 착수 전. 7번의 "결합" 리스크를 실제로 검증하려면 선행 필요
5. **6번(임베딩 스토어)** — 4번이 안정화된 뒤에 착수하는 게 순서상 자연스러움(정제 안 된 벡터를 저장해봤자 재사용 가치 없음)
