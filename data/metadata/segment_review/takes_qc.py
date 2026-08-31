"""전체 테이크 자동 스크리닝 — discarded_takes 채우기 전 1차 확인용.

calibration_check.py와 같은 방식(관절 검출률·프레임 이탈)을 M00 이외의
모든 테이크(video_segments.csv 기준)로 확장. 최종 판단은 사람이 한다 —
여기선 검토 후보만 CSV로 뽑는다.

실행 (repo 루트에서): .venv/bin/python data/metadata/segment_review/takes_qc.py
결과: data/metadata/segment_review/takes_qc_result.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from calibration_check import check_clip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SEGMENTS_CSV = REPO_ROOT / "data" / "metadata" / "video_segments.csv"
OUT_CSV = Path(__file__).parent / "takes_qc_result.csv"

DETECT_THRESHOLD = 0.90
IN_FRAME_THRESHOLD = 0.90


def main() -> None:
    with SEGMENTS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows, 1):
        clip_path = REPO_ROOT / row["file"]
        print(f"[{i}/{len(rows)}] {clip_path.name}", flush=True)
        if not clip_path.exists():
            results.append({**row, "detection_rate": "", "in_frame_rate": "", "verdict": "파일없음"})
            continue
        r = check_clip(clip_path)
        ok = r["detection_rate"] >= DETECT_THRESHOLD and r["in_frame_rate"] >= IN_FRAME_THRESHOLD
        verdict = "OK" if ok else "검토필요"
        results.append({
            **row,
            "detection_rate": f"{r['detection_rate']:.3f}",
            "in_frame_rate": f"{r['in_frame_rate']:.3f}",
            "verdict": verdict,
        })

    fieldnames = list(rows[0].keys()) + ["detection_rate", "in_frame_rate", "verdict"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    flagged = [r for r in results if r["verdict"] != "OK"]
    print(f"\n총 {len(results)}개 중 검토 필요 {len(flagged)}개 → {OUT_CSV}")
    for r in flagged:
        print(f"  {r['participant_id']} {r['action']} T{r['take']}: {r['verdict']} "
              f"(detect={r['detection_rate']}, in_frame={r['in_frame_rate']})")


if __name__ == "__main__":
    main()
