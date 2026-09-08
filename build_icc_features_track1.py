"""트랙 1 실촬영 데이터(data/raw)로 build_icc_features.py와 같은 서명 추출 → npz 저장.

대상: P001~P007 (아직 파일 정리 중인 P008~P015, discarded_takes로 재확인 필요한
P005의 M06/M07은 기본 제외 대상 — [[project-track1-data-status]]) x 동작 하나의 첫 N테이크.
N은 7명 전원이 공통으로 가진 최소 테이크 수에 맞춘다(icc_3_1은 균형설계 필요 —
evaluate_icc.py 참고) — M02는 3(P004가 3테이크뿐), M04도 3(P001/P002/P004가 3테이크뿐).

_nosound 쪽을 쓰는 이유: video_segments.csv·calibration_check.py 등 기존 파이프라인이
이미 그 경로를 정본으로 쓰고 있음(오디오 유무만 다르고 포즈 추출엔 무관).

사용법 (repo 루트에서, mediapipe 설치된 venv로):
  .venv/bin/python build_icc_features_track1.py --motion M02 --takes 3 --joints right_arm
  .venv/bin/python build_icc_features_track1.py --motion M04 --takes 3 --joints hips
  .venv/bin/python build_icc_features_track1.py --motion M06 --takes 2 --joints hips --participants P001,P002,P003,P004,P006,P007
  .venv/bin/python evaluate_icc.py --npz icc_features_track1_M04_mean_speed_hips.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_icc_features import JOINT_PRESETS, build_features, signature

DEFAULT_PARTICIPANTS = "P001,P002,P003,P004,P005,P006,P007"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", default="M02", help="예: M02, M04")
    parser.add_argument("--takes", type=int, default=3, help="사람마다 앞에서부터 몇 테이크 쓸지(공통 최소치)")
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS,
                         help="쉼표구분 participant_id — 재확인 필요한 사람 빼고 싶을 때(예: M06의 P005) 씀")
    parser.add_argument("--feature", choices=["mean_speed", "mean_jerk"], default="mean_speed")
    parser.add_argument("--joints", choices=list(JOINT_PRESETS), default="single")
    parser.add_argument("--out", default=None, help="기본: icc_features_track1_{motion}_{feature}_{joints}.npz")
    args = parser.parse_args()
    joints = JOINT_PRESETS[args.joints]
    out = args.out or f"icc_features_track1_{args.motion}_{args.feature}_{args.joints}.npz"
    participants = args.participants.split(",")

    subjects = {
        p: [f"data/raw/_nosound/{p}_S1_{args.motion}_T{t:02d}_CAM_F.mov" for t in range(1, args.takes + 1)]
        for p in participants
    }
    for name, files in subjects.items():
        for f in files:
            if not Path(f).is_file():
                raise SystemExit(f"파일 없음: {f}")

    print(f"서명 추출 중 ({sum(len(v) for v in subjects.values())}개 영상, {args.motion} 첫 {args.takes}테이크, feature={args.feature}, joints={args.joints})...")
    names, features = build_features(subjects, extract_fn=lambda p: signature(p, args.feature, joints))
    for name, row in zip(names, features):
        print(f"  {name}: {['%.4f' % v for v in row]}")

    np.savez(out, features=features, subjects=np.array(names))
    print(f"\n저장 완료: {out}  (shape={features.shape})")
