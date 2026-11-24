"""Movement Style 파일럿 — 원본 영상 6개에서 MediaPipe 3D를 추출해 npz로 저장한다.

팀 저장소의 mp3d_extract.extract(path)만 그대로 쓴다(main()은 쓰지 않음, 기존 파일 수정 없음).
원본 영상은 읽기만 한다. 출력은 저장소 밖 dynamic_data/processed/mp3d/.

npz 형식은 mp3d_extract.main()과 같다(world, xy, fps). 어떤 원본에서 뽑았는지 확인하려고
source_name·source_size를 함께 저장하고, 이 두 값이 지금 원본과 같으면 다시 추출하지 않는다.

실행 (samsam_pilot conda 환경):
  /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/extract_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKTREE = Path.home() / "Documents" / "dynamic-movement-style"  # 팀 코드(origin/main 7882963), 읽기 전용으로 import
DATA = Path.home() / "Documents" / "dynamic_data"
RAW = DATA / "raw" / "nosound"
OUT = DATA / "processed" / "mp3d"
CLIPS = [f"{p} {m}" for p in ("P001", "P002", "P003") for m in ("Walking", "Jogging")]

sys.path.insert(0, str(WORKTREE))
from mp3d_extract import extract  # noqa: E402


def up_to_date(dst: Path, src: Path) -> bool:
    if not dst.exists():
        return False
    with np.load(dst) as d:
        return "source_name" in d and str(d["source_name"]) == src.name and int(d["source_size"]) == src.stat().st_size


def main() -> None:
    missing = [c for c in CLIPS if not (RAW / f"{c}.mov").exists()]
    if missing:
        raise SystemExit(f"원본 없음: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CLIPS:
        src, dst = RAW / f"{c}.mov", OUT / f"{c}.npz"
        if up_to_date(dst, src):
            print("건너뜀(이미 있음)", dst.name)
            continue
        d = extract(src)
        np.savez(dst, **d, source_name=src.name, source_size=src.stat().st_size)
        print("저장", dst.name, d["world"].shape, f"{d['fps']:.3f}fps")


if __name__ == "__main__":
    main()
