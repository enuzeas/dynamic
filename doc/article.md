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
통계·시계열 도구 (확실히 검증된 것들)
방법론 층에서 라이선스도 깨끗하고 논문도 명확한 것들입니다. DTW와 그 변종(soft-DTW는 미분 가능해서 신경망에 넣기 좋음)은 시계열 정렬의 표준이고, tslearn(MIT) 같은 라이브러리에 구현돼 있습니다. 개인 일관성을 정량화할 때는 반복 신뢰도(test-retest reliability) 통계 — 급내상관계수(ICC) 같은 게 정석입니다. "같은 사람이 반복해도 특징이 일관된가"를 ICC로 재고, "사람 간엔 다른가"를 분류 정확도로 재면, 당신 가설을 두 방향에서 검증하는 깔끔한 실험 설계가 됩니다.
