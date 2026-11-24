"""participant_pairs 검증 + manifest 작성 — STyMo 자체 로더(pymotion BVH)와 STyMo의 features 함수로 다시 읽어 확인한다.

기존 STyMo 코드는 수정하지 않는다(import만 함, `-B`로 pycache를 남기지 않음). 학습·추론은 하지 않는다.

확인 항목 (참가자별, source/target):
  - 파일 이름 동일 / 프레임 수 동일 / frame_time / rot_order / 관절 이름·부모·offset이 STyMo 샘플 스켈레톤과 일치
  - STyMo의 FK 규약(features.py: Root 위치 + Hips 로컬 위치가 offset을 대체)으로 계산한 세계 위치에서
      Root y=0, 발 최저 높이≈0, 머리가 Hips보다 위, 왼쪽 다리가 +X, 발끝이 +Z 방향, NaN 없음
  - STyMo의 get_feet_indices 와 compute_feet_contacts 가 오류 없이 돈다
출력: ~/work/stymo_trial/participant_pairs/manifest.json (이미 있으면 중단)

실행:
  ~/work/stymo_trial/env/bin/python -B ~/work/stymo_trial/tools/check_participant_pairs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

STY = Path.home() / "work" / "stymo_trial"
SRC = STY / "STyMo" / "src"
PAIRS = STY / "participant_pairs"
TEMPLATE = STY / "STyMo" / "motion" / "mocha" / "happy" / "source" / "neutral_to_happy.bvh"
SPEC = {  # 합의된 구간 (구성 스크립트와 같은 값을 여기서 따로 적어 교차 확인)
    "P001": {"source": ["공통_걷기.bvh", [9020, 9200]], "target": ["P001 Walking-M01.npz", [30, 210]]},
    "P002": {"source": ["공통_걷기.bvh", [9032, 9200]], "target": ["P002 Walking.npz", [48, 216]]},
    "P003": {"source": ["공통_걷기.bvh", [9020, 9200]], "target": ["P003 Walking.npz", [228, 408]]},
}

sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))
from pymotion.io.bvh import BVH  # noqa: E402
from pymotion.ops.skeleton_np import fk  # noqa: E402
import features as F  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> BVH:
    b = BVH()
    b.load(str(p))
    return b


def world(b: BVH) -> np.ndarray:
    rot, pos, parents, offsets, _, _ = b.get_data()
    off = np.tile(offsets, (rot.shape[0], 1, 1))
    off[:, 1, :] = pos[:, 1, :]  # features.py: Hips 로컬 위치가 offset을 대체
    p, _ = fk(rot, pos[:, 0, :], off, parents)
    return p


def main() -> None:
    out = PAIRS / "manifest.json"
    if out.exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {out}")
    ref = load(TEMPLATE)
    names = list(ref.data["names"])
    ix = {n: i for i, n in enumerate(names)}
    manifest = {"skeleton": {"template": str(TEMPLATE), "n_joints": len(names), "names": names}, "pairs": {}, "all_ok": True}
    for pid, spec in SPEC.items():
        entry = {"content_source": spec["source"], "style_target": spec["target"], "files": {}, "checks": {}}
        files = {k: sorted((PAIRS / pid / k).glob("*.bvh")) for k in ("source", "target")}
        c = entry["checks"]
        c["one_file_each"] = all(len(v) == 1 for v in files.values())
        c["same_filename"] = c["one_file_each"] and files["source"][0].name == files["target"][0].name
        if not c["same_filename"]:
            entry["ok"] = False; manifest["pairs"][pid] = entry; manifest["all_ok"] = False; continue
        data = {}
        for k in ("source", "target"):
            p = files[k][0]; b = load(p); w = world(b)
            data[k] = (b, w)
            n, ft = int(b.data["rotations"].shape[0]), float(b.data["frame_time"])
            feet = F.get_feet_indices(b)
            contacts = F.compute_feet_contacts(w, feet, ft, 0.6) if True else None
            entry["files"][k] = {"path": str(p), "frames": n, "frame_time": ft, "fps": round(1 / ft, 3), "sha256": sha(p), "bytes": p.stat().st_size,
                                 "feet_joints": [names[i] for i in feet]}
            c[f"{k}_skeleton_matches_template"] = bool(list(b.data["names"]) == names and np.array_equal(b.data["parents"], ref.data["parents"])
                                                       and np.allclose(b.data["offsets"], ref.data["offsets"], atol=1e-6)
                                                       and np.array_equal(b.data["rot_order"], ref.data["rot_order"]))
            c[f"{k}_finite"] = bool(np.isfinite(b.data["rotations"]).all() and np.isfinite(b.data["positions"]).all())
            c[f"{k}_root_y_zero"] = bool(np.abs(b.data["positions"][:, 0, 1]).max() < 1e-6)
            c[f"{k}_feet_min_y_m"] = float(w[:, feet + [ix["LeftToeBase"], ix["RightToeBase"]], 1].min())
            c[f"{k}_head_above_hips"] = bool((w[:, ix["Head"], 1] > w[:, ix["Hips"], 1]).all())
            c[f"{k}_left_leg_plus_x"] = bool((w[:, ix["LeftUpLeg"], 0] > w[:, ix["RightUpLeg"], 0]).all())
            c[f"{k}_toe_forward_plus_z"] = float(((w[:, ix["LeftToeBase"], 2] - w[:, ix["LeftFoot"], 2]).mean()
                                                  + (w[:, ix["RightToeBase"], 2] - w[:, ix["RightFoot"], 2]).mean()) / 2)
            c[f"{k}_hips_height_m_min_max"] = [float(w[:, ix["Hips"], 1].min()), float(w[:, ix["Hips"], 1].max())]
            c[f"{k}_foot_contact_fraction"] = [round(float(np.mean(contacts[:, i])), 3) for i in range(contacts.shape[1])]
        c["frames_equal"] = entry["files"]["source"]["frames"] == entry["files"]["target"]["frames"]
        c["frame_time_equal"] = entry["files"]["source"]["frame_time"] == entry["files"]["target"]["frame_time"]
        want = spec["source"][1][1] - spec["source"][1][0]
        c["frames_match_agreed_range"] = entry["files"]["source"]["frames"] == want == spec["target"][1][1] - spec["target"][1][0]
        flags = [v for k, v in c.items() if isinstance(v, bool)]
        c["feet_min_y_ok"] = all(abs(c[f"{k}_feet_min_y_m"]) < 1e-3 for k in ("source", "target"))
        c["toe_forward_ok"] = all(c[f"{k}_toe_forward_plus_z"] > 0 for k in ("source", "target"))
        entry["ok"] = bool(all(flags) and c["feet_min_y_ok"] and c["toe_forward_ok"])
        manifest["all_ok"] &= entry["ok"]
        manifest["pairs"][pid] = entry
        print(pid, "OK" if entry["ok"] else "FAIL", {k: entry["files"][k]["frames"] for k in entry["files"]}, "fps", entry["files"]["source"]["fps"])
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("manifest:", out, "| all_ok:", manifest["all_ok"])


if __name__ == "__main__":
    main()
