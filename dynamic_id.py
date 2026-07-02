"""BIO-IP Dynamic ID — 등록(enroll) → 저장 → 검증(verify)/식별(identify).

Dynamic ID는 영상 원본이 아니라 관절별 다이내믹스(속도 곡선) 템플릿만 담는다.
scope.md의 "BIO-IP 등록 Mock: 특징점 → 로컬 DB 저장·조회"를 실제로 동작하는
형태로 구현한 것 — 로컬 JSON 파일이 그 "로컬 DB"다.

사용법:
  python dynamic_id.py enroll   --label 이한성 --videos A_1.mp4 A_2.mp4 --out registry/
  python dynamic_id.py verify   --label 이한성 --probe A_3.mp4 --registry registry/ --threshold 40
  python dynamic_id.py identify --probe C_1.mp4 --registry registry/
  python dynamic_id.py calibrate --registry registry/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze import dtw, sweep_far_frr
from demo import trim_motion
from pose_extract import LANDMARK_INDEX, extract_joint_dynamics

DEFAULT_JOINTS = tuple(LANDMARK_INDEX)


@dataclass(frozen=True)
class DynamicID:
    label: str
    joints: tuple[str, ...]
    templates: dict[str, list[list[float]]]  # 관절 -> 등록 반복별 속도 곡선
    created_at: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "joints": list(self.joints),
            "templates": self.templates,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicID":
        return cls(
            label=d["label"],
            joints=tuple(d["joints"]),
            templates=d["templates"],
            created_at=d["created_at"],
        )


def enroll(label: str, video_paths: list[str], joints: tuple[str, ...] = DEFAULT_JOINTS) -> DynamicID:
    """N개 등록 영상에서 관절별 속도 곡선을 뽑아 Dynamic ID 템플릿을 만든다."""
    templates: dict[str, list[list[float]]] = {j: [] for j in joints}
    for vp in video_paths:
        dyn = extract_joint_dynamics(vp, joints)
        for j in joints:
            templates[j].append(trim_motion(dyn[j]["speed"]).tolist())
    return DynamicID(
        label=label,
        joints=joints,
        templates=templates,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_id(id_: DynamicID, registry_dir: str) -> Path:
    Path(registry_dir).mkdir(parents=True, exist_ok=True)
    path = Path(registry_dir) / f"{id_.label}.json"
    path.write_text(json.dumps(id_.to_dict(), ensure_ascii=False, indent=2))
    return path


def load_id(path: str) -> DynamicID:
    return DynamicID.from_dict(json.loads(Path(path).read_text()))


def load_registry(registry_dir: str) -> list[DynamicID]:
    return [load_id(str(p)) for p in sorted(Path(registry_dir).glob("*.json"))]


def _extract_probe(video_path: str, joints: tuple[str, ...]) -> dict[str, np.ndarray]:
    dyn = extract_joint_dynamics(video_path, joints)
    return {j: trim_motion(dyn[j]["speed"]) for j in joints}


def _distance_to_id(probe: dict[str, np.ndarray], id_: DynamicID) -> float:
    """probe와 등록된 반복들 중 가장 가까운 것까지의 DTW 거리, 관절 평균."""
    per_joint = []
    for j in id_.joints:
        reps = [np.array(r, dtype=float) for r in id_.templates[j]]
        per_joint.append(min(dtw(probe[j], r) for r in reps))
    return float(np.mean(per_joint))


def verify(probe_video: str, id_: DynamicID, threshold: float) -> tuple[bool, float]:
    """probe_video가 id_ 본인인지 1:1 판정.
    threshold는 감으로 정하지 말고 calibrate_threshold()로 보정한 값을 쓴다."""
    probe = _extract_probe(probe_video, id_.joints)
    distance = _distance_to_id(probe, id_)
    return distance <= threshold, distance


def identify(probe_video: str, registry: list[DynamicID]) -> tuple[str, float]:
    """probe_video에 가장 가까운 등록자를 1:N 검색."""
    if not registry:
        raise ValueError("registry가 비어 있습니다.")
    probe = _extract_probe(probe_video, registry[0].joints)
    scored = [(id_.label, _distance_to_id(probe, id_)) for id_ in registry]
    scored.sort(key=lambda pair: pair[1])
    return scored[0]


def calibrate_threshold(registry: list[DynamicID]) -> float:
    """등록된 템플릿들의 반복끼리 genuine/impostor 거리로 EER 임계값을 추정한다.
    ponytail: 등록 인원·반복 수가 적으면(예: 인당 2~3회) 추정이 거칠다 —
    scope.md 기준(8~12명, 인당 5회)까지 등록자가 늘면 다시 계산해야 한다."""
    genuine, impostor = [], []
    for id_ in registry:
        for j in id_.joints:
            reps = [np.array(r, dtype=float) for r in id_.templates[j]]
            for i in range(len(reps)):
                for k in range(i + 1, len(reps)):
                    genuine.append(dtw(reps[i], reps[k]))
    for a in range(len(registry)):
        for b in range(a + 1, len(registry)):
            shared_joints = set(registry[a].joints) & set(registry[b].joints)
            for j in shared_joints:
                for ra in registry[a].templates[j]:
                    for rb in registry[b].templates[j]:
                        impostor.append(dtw(np.array(ra), np.array(rb)))

    if not genuine or not impostor:
        raise ValueError("임계값 보정에는 등록자 2명 이상, 각 2회 이상 반복이 필요합니다.")

    thresholds, far_arr, frr_arr = sweep_far_frr(np.array(genuine), np.array(impostor))
    idx = np.argmin(np.abs(far_arr - frr_arr))
    return float(thresholds[idx])


# ── CLI ──────────────────────────────────────────────────

def _cmd_enroll(args: argparse.Namespace) -> None:
    id_ = enroll(args.label, args.videos)
    path = save_id(id_, args.out)
    print(f"등록 완료: {path} (관절 {len(id_.joints)}개, 반복 {len(args.videos)}회)")


def _cmd_verify(args: argparse.Namespace) -> None:
    id_ = load_id(str(Path(args.registry) / f"{args.label}.json"))
    accepted, distance = verify(args.probe, id_, args.threshold)
    verdict = "본인 (accept)" if accepted else "타인 (reject)"
    print(f"{args.label} 대상 검증 — 거리 {distance:.2f} / 임계값 {args.threshold:.2f} → {verdict}")


def _cmd_identify(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    label, distance = identify(args.probe, registry)
    print(f"가장 가까운 등록자: {label} (거리 {distance:.2f}, 등록자 {len(registry)}명 중)")


def _cmd_calibrate(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    threshold = calibrate_threshold(registry)
    print(f"보정된 임계값: {threshold:.2f} (등록자 {len(registry)}명 기준)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--label", required=True)
    p_enroll.add_argument("--videos", nargs="+", required=True)
    p_enroll.add_argument("--out", default="registry")
    p_enroll.set_defaults(func=_cmd_enroll)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--label", required=True)
    p_verify.add_argument("--probe", required=True)
    p_verify.add_argument("--registry", default="registry")
    p_verify.add_argument("--threshold", type=float, required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_identify = sub.add_parser("identify")
    p_identify.add_argument("--probe", required=True)
    p_identify.add_argument("--registry", default="registry")
    p_identify.set_defaults(func=_cmd_identify)

    p_calibrate = sub.add_parser("calibrate")
    p_calibrate.add_argument("--registry", default="registry")
    p_calibrate.set_defaults(func=_cmd_calibrate)

    parsed = parser.parse_args()
    parsed.func(parsed)
