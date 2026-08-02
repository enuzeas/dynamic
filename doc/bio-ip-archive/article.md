1. 생체역학·운동제어 — "개인은 왜 일관된가"
여기가 학술적 뿌리입니다. 운동제어 분야에는 motor signature(운동 서명) 또는 individual motor style이라는 확립된 개념이 있습니다. 같은 과제를 반복해도 사람마다 일관된 운동 패턴을 보인다는 걸 실험으로 보여온 분야죠.
특히 찾아볼 만한 두 개념이 있습니다. 하나는 kinematic individuality / movement signature — 걸음걸이나 제스처에서 개인 고유성을 측정한 연구들. 다른 하나는 minimum jerk theory(최소 저크 이론) 인데, 인간의 자연스러운 움직임이 저크를 최소화하는 궤적을 따른다는 고전 이론입니다(Flash & Hogan 계열). 앞서 저크를 특징으로 쓰자고 한 게 여기에 이론적 근거가 있습니다. "개인마다 이 최소화의 방식이 미묘하게 다르다"가 당신 가설의 출발점이 됩니다.
검색어: motor signature individuality, movement kinematics individual differences, minimum jerk trajectory, inter-trial variability motor control.
2. 보행·제스처 인식 — "움직임으로 사람을 식별한다"
이미 산업화된 분야입니다. gait recognition(보행 인식) 은 CCTV 보안 연구로 수십 년 축적돼 있고, "걸음걸이만으로 신원 식별"이 실제로 됩니다. 당신이 하려는 건 이걸 걸음걸이에서 임의의 퍼포먼스 동작으로 확장하는 것이라, 방법론을 거의 그대로 빌려올 수 있습니다.
기술적으로 이 분야의 표준 도구가 앞서 말한 ST-GCN(Spatio-Temporal Graph Convolutional Network) 과 그 후속(2s-AGCN, MS-G3D 등)입니다. 골격 키포인트 시퀀스를 그래프로 보고 시공간 패턴을 학습하죠. NTU RGB+D라는 대규모 골격 동작 데이터셋이 이 분야 벤치마크라, 방법론·코드를 참고하기 좋습니다.
검색어: skeleton-based gait recognition, ST-GCN action recognition, skeleton based person identification, NTU RGB+D benchmark.
여기서 중요한 구분 하나. 보행 인식은 보통 "동작 종류가 같을 때(다 걷기)" 사람을 맞힙니다. 당신의 주제는 한 발 더 나아갑니다 — 동작 종류와 무관하게(action-invariant) 그 사람의 움직임 스타일만 뽑아낼 수 있는가. 이건 아래 3번과 연결됩니다.
3. 스타일·콘텐츠 분리 — "무엇을 하는가 vs 누가 하는가"
이게 당신 주제의 가장 깊은 층이고, 한우진의 "1인 30역" 질문("역할이 바뀌어도 내 서명은 남는가")이 정확히 여기 해당합니다. 동작에는 콘텐츠(무슨 동작인가) 와 스타일(누가/어떻게 하는가) 이 섞여 있는데, 이 둘을 분리(disentangle)하려는 연구 갈래입니다.
컴퓨터 그래픽스의 motion style transfer(모션 스타일 전이) 가 대표적입니다. "걷기"라는 콘텐츠는 두고 "슬픈 스타일 / 씩씩한 스타일"만 갈아끼우는 기술인데, 이걸 학습한다는 건 곧 스타일을 콘텐츠와 분리해 인코딩한다는 뜻입니다. 그 분리된 스타일 벡터가 바로 당신이 찾는 "다이내믹스 서명"의 후보입니다. 최근엔 이걸 contrastive learning이나 VAE 기반으로 푸는 흐름이 강합니다.
검색어: motion style transfer, disentangling style and content motion, motion style encoding, contrastive learning motion representation.
4. 최근 논문으로 본 실증 현황 (2024–2025)
3번 가설 — "동작이 달라도 그 사람의 스타일만 뽑아낼 수 있는가" — 은 이미 최근 논문들이 실험으로 다루고 있습니다.

**동작 무관 개인 식별을 정면으로 실험한 논문**: DisMo(Disentangled Motion Representations for Open-World Motion Transfer, arXiv 2511.23428, 2025)는 "Identity Reconstruction from Motion Representations" 섹션에서 모션 표현이 동작 내용과 개인 고유 특성을 얼마나 분리하는지 identity classification 실험으로 직접 검증합니다. ID-MotionNet(PRCV 2025)도 정보병목(Information Bottleneck) 기법으로 스켈레톤 시퀀스에서 신원 정보를 동작과 분리해 보존하는 걸 목표로 합니다.

**동작 종류가 달라도 사람을 재식별(re-ID)하는 계열**: Skeletons Speak Louder than Text(arXiv 2511.13150, 2025), Motif Guided Graph Transformer(arXiv 2412.09044, 2024), Rao/Leung/Miao의 Hierarchical Skeleton Meta-Prototype Contrastive Learning(IJCV 2024) 등. gait recognition을 "걷기"에서 임의 동작으로 일반화하려는 흐름과 정확히 맞닿아 있고, 2024~2025에 활발합니다.

**스타일-콘텐츠 분리(모션 스타일 전이)의 최신판**: MoST(Motion Style Transformer between Diverse Action Contents, CVPR 2024)가 이름 그대로 "다른 동작 콘텐츠 사이에서" 스타일을 옮기는 문제를 정면으로 다룹니다. MCM-LDM(2024, trajectory-content-style 3분해), StyleMotif(arXiv 2503.21775, 2025, 멀티모달), PersonaAnimator/PersonaBooth(2025, 개인화 모션 생성)도 같은 흐름.

**저크 자체를 개인 식별 지표로 쓴 논문 — 여전히 공백**: 가장 근접한 논문(Hirose et al., PeerJ 2020, "Integrated jerk as an indicator of affinity for artificial agent kinematics")도 저크를 "인간다움/선호도" 지표로 썼지 개인 식별 지표로 검증한 게 아닙니다. 1번의 minimum-jerk 이론을 개인차 근거로 연결하는 부분은 여전히 가설 단계이며, 대부분의 최근 연구는 손수 설계한 저크 같은 물리량이 아니라 학습된 임베딩(스타일 벡터)을 씁니다. 저크를 명시적 feature로 쓰는 접근은 채워지지 않은 틈새로 보입니다.

검색어: motion style transformer diverse action content, identity disentanglement motion representation, skeleton-based person re-identification, information bottleneck skeleton identity.

참고 링크:

- [DisMo: Disentangled Motion Representations for Open-World Motion Transfer](https://arxiv.org/pdf/2511.23428)
- [ID-MotionNet: Identity-Preserved 3D Skeleton Sequence Generation via Information Bottleneck Disentanglement](https://link.springer.com/chapter/10.1007/978-981-95-5628-1_22)
- [Skeletons Speak Louder than Text: A Motion-Aware Pretraining Paradigm for Video-Based Person Re-Identification](https://arxiv.org/pdf/2511.13150)
- [Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification](https://arxiv.org/pdf/2412.09044)
- [PersonaAnimator: Personalized Motion Transfer from Unconstrained Videos](https://arxiv.org/pdf/2508.19895)
- [Integrated jerk as an indicator of affinity for artificial agent kinematics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7500322/)

5. 통계·시계열 도구 (확실히 검증된 것들)
방법론 층에서 라이선스도 깨끗하고 논문도 명확한 것들입니다. DTW와 그 변종(soft-DTW는 미분 가능해서 신경망에 넣기 좋음)은 시계열 정렬의 표준이고, tslearn(MIT) 같은 라이브러리에 구현돼 있습니다. 개인 일관성을 정량화할 때는 반복 신뢰도(test-retest reliability) 통계 — 급내상관계수(ICC) 같은 게 정석입니다. "같은 사람이 반복해도 특징이 일관된가"를 ICC로 재고, "사람 간엔 다른가"를 분류 정확도로 재면, 당신 가설을 두 방향에서 검증하는 깔끔한 실험 설계가 됩니다.

6. 구현 이후 드러난 보완점 논문 (pose_extract.py · dynamic_id.py 기준)
실제로 MediaPipe PoseLandmarker 기반 추출 파이프라인(pose_extract.py)과 등록·검증·식별 모듈(dynamic_id.py)을 코드로 짜서 샘플 영상에 돌려보니, 1:N 식별(identify)은 맞았지만 1:1 검증(verify)의 임계값 판정은 등록 표본이 너무 적어(2명×2회) 아직 못 미더운 상태로 나왔습니다. 이 네 가지 구체적 약점에 각각 대응하는 문헌입니다.

**① 템플릿이 "가장 가까운 반복 하나"에만 의존한다 — DTW Barycenter Averaging(DBA)**: 지금 `_distance_to_id()`는 등록 반복 중 최근접 하나와만 비교해 노이즈에 취약합니다. Petitjean·Ketterlin·Gançarski의 원조 DBA(2011)와 tslearn의 `dtw_barycenter_averaging` 구현을 쓰면 등록 반복 여러 개를 DTW로 정렬해 대표 곡선 하나로 합칠 수 있습니다. 수렴성을 다룬 최신 후속 연구도 있습니다(arXiv 2401.05841, 2024).

**② `calibrate_threshold()`가 소표본(등록 2명×2회)에서 거칠다 — few-shot 임계값 보정**: "Self-Learning for Personalized Keyword Spotting on Ultra-Low-Power Audio Sensors"(arXiv 2408.12481, 2024)가 거의 같은 문제를 다룹니다 — 등록 시 극소 샘플(K=3)만으로 프로토타입 벡터를 만들고 positive/negative 거리 마진을 최대화하는 임계값을 고르는 알고리즘. 지금 코드가 하는 genuine/impostor 거리 스윕의 소표본 정공법입니다.

**③ 관절별 거리를 단순 평균으로 융합한다 — score-fusion 문헌**: gait+keystroke 융합에 신뢰도 기반 가중치를 쓰는 Context-Driven Multi-Biometric Scoring Algorithm(CMBSA, Scientific Reports 2025, EER 2.35%)이나 dual-attention transformer 기반 융합(EER 2.3%)은, 관절마다 신호 품질이 다르다는 걸(손목이 어깨보다 서명이 강함) 가중치로 반영하는 근거가 됩니다.

**④ MediaPipe 지터가 저크·가속도로 그대로 전파된다 — 스무딩 필터 재검토**: MediaPipe 공식 GitHub 이슈(google/mediapipe#4507)에서 Tasks API로 넘어오며 legacy에 있던 스무딩 파라미터가 빠졌다고 개발자들이 인정하고 있고, 커뮤니티는 고정 윈도우 Savitzky-Golay보다 속도에 비례해 스무딩 강도를 조절하는 One-Euro 필터나 Kalman을 더 권합니다. 빠른 팔 동작 구간에서 과도하게 뭉개지는 걸 막는 다음 개선점입니다.

우선순위: 지금 가장 싸게 정확도를 올릴 수 있는 건 ①One-Euro 필터 교체와 ④DBA 템플릿 두 가지고, ②few-shot 보정·③score-fusion은 등록자가 늘어난 뒤(plan.md Phase 3 이후) 의미가 커집니다.

검색어: DTW barycenter averaging, few-shot threshold calibration biometric, multi-modal score fusion gait keystroke, MediaPipe pose landmark jitter smoothing.

참고 링크:

- [dtw_barycenter_averaging — tslearn documentation](https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html)
- [On the number of iterations of the DBA algorithm](https://arxiv.org/pdf/2401.05841)
- [Self-Learning for Personalized Keyword Spotting on Ultra-Low-Power Audio Sensors](https://arxiv.org/pdf/2408.12481)
- [Enhancing security and usability with context aware multi-biometric fusion for continuous user authentication](https://www.nature.com/articles/s41598-025-14833-z)
- [Pose Landmarker Jittering · Issue #4507 · google-ai-edge/mediapipe](https://github.com/google/mediapipe/issues/4507)
- [Setting-up Smoothing Filters for MediaPipe Pose Estimation Pipeline (One-Euro/Kalman)](https://medium.com/@debasishraut.dev/setting-up-smoothing-filters-for-mediapipe-pose-estimation-pipeline-a-practical-guide-fcc03f462196)
