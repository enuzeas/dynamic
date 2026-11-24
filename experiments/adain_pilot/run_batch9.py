"""Movement Style batch9 — 절대 AdaIN 9개 조건(다른 사람 6 + 같은 사람 3)을 기존 run_style_pilot.main()으로 실행한다.

run_style_pilot.py는 수정하지 않는다. 이 프로세스 안에서만 모듈 상수를 바꾼다:
  PILOTS -> 9개 (스타일 주인, 콘텐츠 주인)
  WALK, JOG -> P003 구간 추가 (P001·P002 값은 기존과 같음)
  OUT -> movement_style_batch9_absolute/ (Pilot 1·2 폴더 movement_style_pilot/은 건드리지 않음)
AdaIN 조건(absolute, group=None, frontal_sd=False)과 처리 순서는 run_style_pilot.main() 그대로다.

조건 폴더가 하나라도 이미 있으면 덮어쓰지 않고 시작 전에 멈춘다.
실행 후 manifest.json에 조건·구간·코드 커밋·환경 버전·스크립트 SHA-256·시각을 남긴다.

실행:
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/run_batch9.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_style_pilot as X  # noqa: E402  (NUMPY_EXPERIMENTAL_DTYPE_API=1 검사도 여기서 한다)
import reference_metrics as RM  # noqa: E402

PEOPLE = ("P001", "P002", "P003")
CROSS = [(s, c) for s in PEOPLE for c in PEOPLE if s != c]
SAME = [(p, p) for p in PEOPLE]
CONDITIONS = CROSS + SAME
OUT = RM.OUT


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    if str(OUT.resolve()).startswith(str(X.WORKTREE.resolve())):
        raise SystemExit(f"저장소 안 경로 금지: {OUT}")
    existing = [f"pilot_{s}style_{c}content" for s, c in CONDITIONS if (OUT / f"pilot_{s}style_{c}content").exists()]
    if existing or (OUT / "manifest.json").exists():
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {existing or 'manifest.json'}")

    X.PILOTS = CONDITIONS
    X.WALK = {p: RM.RANGES[p]["Walking"] for p in PEOPLE}
    X.JOG = {p: RM.RANGES[p]["Jogging"] for p in PEOPLE}
    X.OUT = OUT
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    X.main()

    import numpy as np
    commit = subprocess.run(["git", "-C", str(X.WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "started": started,
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "conditions": {"cross_person": [list(c) for c in CROSS], "same_person": [list(c) for c in SAME]},
        "frames": RM.RANGES,
        "adain": {"mode": "absolute", "group": None, "frontal_sd": False},
        "body_space": "CMU 31 (style_compare.ik -> bvh_points)",
        "team_code": {"worktree": str(X.WORKTREE), "commit": commit},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable},
        "scripts_sha256": {p.name: sha256(p) for p in (HERE / "run_batch9.py", HERE / "reference_metrics.py",
                                                       HERE / "run_style_pilot.py", X.WORKTREE / "style_compare.py")},
        "references_json": "references.json (reference_metrics.py로 따로 생성)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("저장", OUT / "manifest.json")


if __name__ == "__main__":
    main()
