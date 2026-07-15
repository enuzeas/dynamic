# BIO-IP 관련 학술논문 목록

---

## A. 운동제어 · 개인 운동 서명 (Motor Control & Motor Signature)

| # | 논문 | 저널/출처 | 핵심 |
|---|---|---|---|
| A1 | Flash, T. & Hogan, N. (1985). **The coordination of arm movements: an experimentally confirmed mathematical model.** | *Journal of Neuroscience* 5(7):1688–1703 | 최소 저크 이론 원전. 인간 팔 운동의 궤적이 저크를 최소화한다는 수학적 모델. |
| A2 | Huh, D. & Sejnowski, T.J. (2015). **Spectrum of power laws for curved hand movements.** | *PNAS* | 2/3 power law — 곡선 운동의 속도-곡률 관계. 최소 저크 이론 확장. |
| A3 | Viviani, P. & Flash, T. (1995). **Minimum-Jerk, Two-Thirds Power Law, and Isochrony: Converging Approaches to Movement Planning.** | *Journal of Experimental Psychology: HPP* | 운동 계획의 세 이론이 수렴한다는 것을 보여줌. |
| A4 | **Movement vigor as a trait-like attribute of individuality.** | *PubMed* PMID: 29766769 | 운동 속도(vigor)가 개인의 특질처럼 일관되게 유지된다는 실험 증거. |
| A5 | **Decoding identity from motion: how motor similarities colour our perception of self and others.** | *ResearchGate* | 움직임 패턴으로 자타 정체성을 인식하는 인지 메커니즘. |
| A6 | **Doing It Your Way: How Individual Movement Styles Affect Action Prediction.** | *PubMed* PMID: 27780259 | 개인 운동 스타일이 타인의 행동 예측에 미치는 영향. |
| A7 | **Discovering individual-specific gait signatures from data-driven models of neuromechanical dynamics.** | *PMC* PMC10610102 | 신경역학 모델에서 개인 고유 보행 서명 추출. |

---

## B. 골격 기반 행동 인식 · 보행 식별 (Skeleton-based Recognition & Gait Identification)

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| B1 | Yan, S. et al. (2018). **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.** | [arxiv:1801.07455](https://arxiv.org/abs/1801.07455) | **ST-GCN 원전.** 골격 키포인트 시퀀스를 시공간 그래프로 모델링. 이 분야 표준 방법론. |
| B2 | **Gait Identification under Surveillance Environment based on Human Skeleton.** | [arxiv:2111.11720](https://arxiv.org/abs/2111.11720) | ST-GCN 기반 보행 표현 학습으로 감시 환경에서 개인 식별. |
| B3 | Teepe, T. et al. (2022). **Towards a Deeper Understanding of Skeleton-based Gait Recognition.** | [arxiv:2204.07855](https://arxiv.org/abs/2204.07855) | 골격 기반 보행 인식의 방법론 분석 — ST-GCN 적용 정리. |
| B4 | **GaitRef: Gait Recognition with Refined Sequential Skeletons.** | [arxiv:2304.07916](https://arxiv.org/abs/2304.07916) | 정제된 골격 시퀀스로 보행 인식 정확도 향상. |
| B5 | **Condition-Adaptive Graph Convolution Learning for Skeleton-Based Gait Recognition.** | [arxiv:2308.06707](https://arxiv.org/abs/2308.06707) | 조건 적응형 그래프 컨볼루션 — 가변 조건에서 골격 보행 인식. |
| B6 | **GaitPT: Skeletons Are All You Need For Gait Recognition.** | [arxiv:2308.10623](https://arxiv.org/abs/2308.10623) | 골격만으로 보행 인식이 충분하다는 주장. Transformer 기반. |

---

## C. 모션 스타일 전이 · 스타일-콘텐츠 분리 (Motion Style Transfer & Disentanglement)

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| C1 | **VQ-Style: Disentangling Style and Content in Motion with Residual Quantized Representations.** | [arxiv:2602.02334](https://arxiv.org/abs/2602.02334) | RVQ-VAE로 모션의 콘텐츠(거친 속성)와 스타일(미세 표현)을 분리. 가장 최신. |
| C2 | Jang, D. et al. (2022). **Motion Puzzle: Arbitrary Motion Style Transfer by Body Part.** | [arxiv:2202.05274](https://arxiv.org/abs/2202.05274) | 신체 부위별 스타일 전이. 콘텐츠-스타일 분리를 신체 분절 단위로 적용. |
| C3 | **Generative Human Motion Stylization in Latent Space.** | [arxiv:2401.13505](https://arxiv.org/abs/2401.13505) | 잠재 공간에서 인간 모션 스타일화. |
| C4 | **StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion.** | [arxiv:2503.21775](https://arxiv.org/abs/2503.21775) | 멀티모달 입력에서 스타일-콘텐츠 크로스 퓨전. |
| C5 | **AStF: Motion Style Transfer via Adaptive Statistics Fusor.** | [arxiv:2511.04192](https://arxiv.org/abs/2511.04192) | 적응형 통계 퓨저로 모션 스타일 전이. |
| C6 | **D-LORD for Motion Stylization.** | [arxiv:2412.04097](https://arxiv.org/abs/2412.04097) | 모션 스타일화를 위한 분리 표현 학습. |

**2026-07-15 후속 검색 — 중요 구분: motion-to-motion vs text-to-motion.** 아래 두 그룹은 표면적으로 다 "모션 스타일 전이"지만 입력이 다르다. 쌤쌤이 필요한 건 "실제로 촬영한 이 콘텐츠 동작을 그대로 유지하며 스타일만 바꾸는" **motion-to-motion**이지, "텍스트로 콘텐츠를 지정하고 스타일만 얹는" **text-to-motion+style**이 아니다. 후자는 화려하고 최신(2024~2026 diffusion 붐)이지만 콘텐츠가 텍스트 프롬프트에서 diffusion이 상상해낸 것이라 특정 인물의 캡처 동작을 그대로 보존하지 못한다 — 쌤쌤 코어("같은 동작이 사람마다 다른 결로 움직인다")에 안 맞는다.

| # | 논문 | 출처 | 콘텐츠 입력 | 핵심 |
|---|---|---|---|---|
| C7 | **VQ-Style: Disentangling Style and Content in Motion with Residual Quantized Representations.** (Disney Research, 2026) | [arxiv:2602.02334](https://arxiv.org/html/2602.02334v2) | **모션** (콘텐츠 클립을 인코더로 코드화 후 style code만 교체 — Quantized Code Swapping) | motion-to-motion. Disney Research 소속 확인 — 산업급 신뢰도. 공개 코드 미확인(2026-07-15 기준) — 재구현 필요할 수 있어 6회 캡스톤엔 리스크. C1과 동일 논문(우선순위 갱신). |
| C8 | **Scalable Motion Style Transfer with Constrained Diffusion Generation.** | [arxiv:2312.07311](https://arxiv.org/abs/2312.07311) | **모션** (source domain keyframe을 content constraint로 사용) | motion-to-motion diffusion. Motion Puzzle(AdaIN)의 diffusion판 대안 — text-to-motion 함정에 안 걸리는 유일한 diffusion 후보. 코드 공개 여부 미확인, 다음 스파이크에서 확인 필요. |
| C9 | **SMooDi: Stylized Motion Diffusion Model.** (ECCV 2024) | [arxiv:2407.12783](https://arxiv.org/abs/2407.12783) — [코드](https://neu-vi.github.io/SMooDi/) | **텍스트** (사전학습 text-to-motion 모델 + style motion 1개로 가이드) | text-to-motion+style. 공개 코드 있음, 인용 많음 — 그러나 콘텐츠가 텍스트발이라 쌤쌤 코어 파이프라인엔 부적합. "텍스트로 새 동작 생성 + 스타일"이라는 인접 기능(나중 확장) 후보로만 기록. |
| C10 | **Stylized Text-to-Motion Generation via Hypernetwork-Driven Low-Rank Adaptation.** (2026-05, 최신) | [arxiv:2605.13333](https://arxiv.org/abs/2605.13333) | **텍스트** | text-to-motion+style. SMooDi 계열의 최신형(hypernetwork가 style 예시 1개→LoRA 파라미터 매핑, 미학습 스타일 일반화 우수). 역시 콘텐츠는 텍스트발 — C9과 같은 이유로 코어 파이프라인엔 부적합. |
| C11 | **Dance Like a Chicken: Low-Rank Stylization for Human Motion Diffusion (LoRA-MDM).** | [arxiv:2503.19557](https://arxiv.org/abs/2503.19557) | **텍스트** | text-to-motion+style. LoRA로 스타일당 소수 샘플만으로 가볍게 파인튜닝 — 컴퓨트는 가볍지만 콘텐츠 입력이 텍스트라 C9·C10과 같은 한계. |
| C12 | **MulSMo: Multimodal Stylized Motion Generation by Bidirectional Control Flow.** | [arxiv:2412.09901](https://arxiv.org/abs/2412.09901) | **텍스트** (content prompt) | text-to-motion+style, 스타일 입력만 텍스트/이미지/모션으로 멀티모달 확장. 콘텐츠는 여전히 텍스트발. |
| C13 | **Diffusion-based Human Motion Style Transfer with Semantic Guidance.** (CGF 2024) | [arxiv:2405.06646](https://arxiv.org/abs/2405.06646) | **텍스트** (1단계: text-to-motion prior 사전학습, 2단계: style 예시 1개로 파인튜닝) | text-to-motion+style. "Style transfer"라는 이름이지만 콘텐츠는 텍스트발 — C9·C10·C11과 같은 그룹. |

**결론**: motion-to-motion 축에서 실전 후보는 여전히 **Motion Puzzle(2022, 검증된 공개 코드)**이 1순위. VQ-Style(C7)·Constrained Diffusion(C8)은 더 최신·고품질이지만 공개 코드 미확인이라 지금 스파이크의 확정 대안이 아니라 "다음 스파이크에서 코드 유무부터 확인할 후보"로만 잡는다.

---

## H. 풋스케이트 보정 · 최신 단안 모션캡처 (Footskate Correction & Latest Monocular MoCap, 2026-07-15 검색)

카이스트 교수 상담 질문 1("단안 RGB로 스타일 캡처가 깨짐(footskate) 없이 나오는 게 현실적인가")에 대한 최신 답.

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| H1 | **FootMR: Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture.** | [arxiv:2603.09681](https://arxiv.org/html/2603.09681) | 2D 발 keypoint(엄지·새끼발가락·뒤꿈치·발목)만 입력받아 잔차 발목 회전을 예측하는 transformer. 이미지 재학습 없이 **GVHMR 같은 기존 3D 복원 모델 위에 붙이는 후처리**로 발목 각도 오차 최대 30% 감소, 학습에 없던 극단적 발 포즈에도 일반화. "학습 아니라 조립"이라는 팀 원칙에 정확히 맞는 형태의 부품. |
| H2 | **RAM: Recover Any 3D Human Motion in-the-Wild.** (CVPR 2026) | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Jia_RAM_Recover_Any_3D_Human_Motion_in-the-Wild_CVPR_2026_paper.html) | 실시간·정확도를 겨냥한 최신 단안 3D 모션 복원 프레임워크 — WHAM/GVHMR 다음 세대 후보. |
| H3 | **OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery.** | [arxiv:2603.17355](https://arxiv.org/pdf/2603.17355) | 온라인(스트리밍) 방식의 world-grounded 인체 메시 복원 — 실시간 파이프라인 필요해질 때 참고. |
| H4 | **Skinned Motion Retargeting with Spatially Adaptive Interaction Guidance.** | [arxiv:2605.19355](https://arxiv.org/pdf/2605.19355) | 접촉·관통(penetration)을 고려한 리타겟팅 — 3회차(SMPL→Mixamo) 리타겟에서 footskate/관통이 나올 경우의 업그레이드 후보. |

**정리**: H1(FootMR)은 지금 스파이크에 바로 붙여볼 수 있는 저비용 부품이라 우선순위가 가장 높다. H2·H3·H4는 "나중에, 필요해지면" 후보로 기록만 해둔다.

---

## D. 행동 기반 생체인식 (Behavioral Biometrics)

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| D1 | Yampolskiy, R.V. & Govindaraju, V. (2008). **Behavioural biometrics: a survey and classification.** | *IJBM* | **행동 생체인식 분류 체계 정립.** 서베이 논문으로 분야 개요 파악에 적합. |
| D2 | **Comprehensive Survey: Biometric User Authentication Application, Evaluation, and Discussion.** | [arxiv:2311.13416](https://arxiv.org/abs/2311.13416) | 생체인증 전반 최신 서베이. EER·rank-1 accuracy 지표 설명 포함. |
| D3 | **A behaviour biometrics dataset for user identification and authentication.** | [PMC9679689](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9679689/) | 행동 생체인식 데이터셋 구축 방법론 — 수집 설계 참고용. |

---

## E. 시계열 정렬 · DTW (Time Series Alignment)

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| E1 | Sakoe, H. & Chiba, S. (1978). **Dynamic programming algorithm optimization for spoken word recognition.** | *IEEE TASLP* | **DTW 원전.** 음성 인식에서 최초 제안. |
| E2 | Salvador, S. & Chan, P. (2007). **FastDTW: Toward accurate dynamic time warping in linear time and space.** | *KDD Workshop* | DTW 이차 복잡도 → 선형으로 단축. 대용량 시계열에 필수. |
| E3 | Cuturi, M. (2017). **Soft-DTW: a Differentiable Loss Function for Time-Series.** | *ICML* | 미분 가능한 DTW — 신경망에 시계열 손실로 직접 삽입 가능. |
| E4 | **An Empirical Evaluation of Similarity Measures for Time Series Classification.** | [arxiv:1401.3973](https://arxiv.org/abs/1401.3973) | DTW와 다른 유사도 측정치 비교 벤치마크. |
| E5 | **DTW Based Verification Scheme of Biometric Signatures.** | [ResearchGate](https://www.researchgate.net/publication/224712969) | DTW를 생체 서명 검증에 적용한 실험. |

---

## F. 검증 통계 · 실험 설계 (Validation Statistics)

| # | 논문 | 출처 | 핵심 |
|---|---|---|---|
| F1 | Koo, T.K. & Li, M.Y. (2016). **A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research.** | *Journal of Chiropractic Medicine* | **ICC 사용 가이드라인.** 항상성 검증에서 어떤 ICC 모델을 써야 하는지 명시. |
| F2 | Shrout, P.E. & Fleiss, J.L. (1979). **Intraclass correlations: Uses in assessing rater reliability.** | *Psychological Bulletin* | ICC 원전. 반복 측정 신뢰도 계산의 기준. |
| F3 | Warrens, M.J. (2015). **Five Ways to Look at Cohen's Kappa.** | *Journal of Psychology & Psychotherapy* | Cohen's Kappa — 분류기 간 일치도 보조 지표. |

---

## G. 미세 움직임 포착 (Subtle / Micro-Movement Capture)

"작은 관절·작은 동작이 신호로서 의미가 있는가"에 답하기 위해 찾은 논문들.

| # | 논문 | 저널/출처 | 핵심 |
|---|---|---|---|
| G1 | Wu, H.-Y. et al. (2012). **Eulerian Video Magnification for Revealing Subtle Changes in the World.** | *ACM TOG* (SIGGRAPH 2012) — [PDF](https://people.csail.mit.edu/mrub/papers/vidmag.pdf) | 육안으로 안 보이는 미세한 색상·움직임 변화(맥박, 호흡)를 시공간 필터링으로 증폭. `dynamics_layer.md`의 "호흡 미세 흔들림" 신호를 실제로 뽑아내는 데 쓸 수 있는 원천 기법. |
| G2 | Cornman, H.L., Stenum, J., Roemmich, R.T. (2021). **Video-based quantification of human movement frequency using pose estimation: A pilot study.** | *PLoS One* 16(12) | OpenPose로 손가락 태핑 등 반복 소동작 5종의 주파수를 측정 — 이벤트 타이밍은 r>0.99로 정확했으나, 표본 10명·모션캡처 대조군 없음이 한계. monocular pose estimation이 손가락 수준 소동작을 어디까지 신뢰성 있게 잡는지에 대한 직접적 파일럿 증거. |
| G3 | Gionfrida, L., Rusli, W.M.R., Bharath, A.A., Kedgley, A.E. (2022). **Validation of two-dimensional video-based inference of finger kinematics with pose estimation.** | *PLOS ONE* 17(11):e0276799 | OpenPose 손가락 keypoint를 마커 기반 모션캡처(골드 스탠다드)와 정량 대조. 단안 카메라 기반 손가락 kinematics 추정의 정확도 한계를 수치로 제시 — 관절 확장 시 어느 정도 오차를 감수해야 하는지 참고. |
| G4 | Liu, X. et al. (2021). **iMiGUE: An Identity-free Video Dataset for Micro-Gesture Understanding and Emotion Analysis.** | *CVPR 2021* — [arXiv:2107.00285](https://arxiv.org/abs/2107.00285) | 무의식적 소동작(micro-gesture)만으로 감정을 분석하되 얼굴·신원 정보는 의도적으로 배제("identity-free"). 소동작을 신원과 분리 가능한 것으로 설계한 전제가, BIO-IP의 "다이내믹스=개인 고유 신원 신호"라는 주장과 정반대 방향이라 비교 검토할 가치가 있음. |
| G5 | Chen, H. et al. (2023). **SMG: A Micro-gesture Dataset Towards Spontaneous Body Gestures for Emotional Stress State Analysis.** | *IJCV* (FG 2019 확장판) | 저강도 자발적 소동작을 Kinect 골격 데이터(4모달리티)로 캡처해 심리적 스트레스 상태와의 상관관계를 분석. 소동작에서 골격 기반 특징을 뽑는 파이프라인 설계 참고용. |
| G6 | Guo, D. et al. (2024). **Benchmarking Micro-action Recognition: Dataset, Methods, and Applications.** (MA-52) | [arXiv:2403.05234](https://arxiv.org/abs/2403.05234), *IEEE TCSVT* | 심리 인터뷰에서 전신 소동작(손-몸통, 머리-손, 다리 상호작용 포함)을 대규모(22K 샘플)로 수집. 손목 이하 국소 관절이 아니라 신체 부위 간 상호작용 단위로 "작은 동작"을 정의하는 관점 — 관절 선택 논의에 참고. |

**정리**: G2·G3는 "손가락 단위 소동작도 근접 촬영·통제된 조건이면 pose estimation으로 꽤 정확히 잡힌다"는 증거지만, 둘 다 손을 화면 가까이 크게 잡은 임상/실험 세팅이다. 지금 프로젝트의 전신 댄스 챌린지 촬영 거리에서는 이 정확도가 그대로 옮겨오지 않는다 — G1(Eulerian magnification)이 오히려 "먼 거리에서도 미세 신호를 증폭해서 잡는" 대안 기법으로 더 맞을 수 있음. G4·G6은 소동작을 다루는 기존 연구들이 대부분 감정·심리 분석 목적이라 "신원 식별"이 아닌 다른 각도에서 소동작을 본다는 점도 유의.

---

## 우선순위 요약

| 우선도 | 논문 | 이유 |
|---|---|---|
| ★★★ | A1 (Flash & Hogan 1985) | 명제의 이론적 뿌리. 반드시 인용. |
| ★★★ | B1 (ST-GCN 2018) | 파이프라인 핵심 방법론. |
| ★★★ | F1 (ICC 가이드라인) | 항상성 검증 지표 설계 근거. |
| ★★★ | D2 (Biometric Survey) | EER·rank-1 지표 정당화. |
| ★★★ | H1 (FootMR, 2026) | footskate 리스크에 대한 저비용 즉시 적용 답. 교수 질문 1 직접 대응. |
| ★★ | C2 (Motion Puzzle) | motion-to-motion 스타일 전이 1순위 후보 — 검증된 공개 코드. |
| ★★ | C1/C7 (VQ-Style) | 스타일-콘텐츠 분리 최신 방법론(Disney Research). 코드 공개 확인 후 격상 검토. |
| ★★ | A4 (Movement vigor) | 고유성 명제의 실험 증거. |
| ★★ | B3 (Skeleton Gait) | 골격 기반 식별 방법론 정리. |
| ★ | E3 (Soft-DTW) | 신경망 통합 시 필요. |
| ★ | C8 (Constrained Diffusion) | motion-to-motion diffusion 대안 — 코드 유무 확인 필요. |
| — | C9–C13 (SMooDi·LoRA-MDM·Hypernetwork-LoRA·MulSMo·Semantic Guidance) | text-to-motion+style 계열 — 쌤쌤 코어 파이프라인엔 부적합, 텍스트 기반 신규 동작 생성이라는 별도 기능으로만 기록. |
