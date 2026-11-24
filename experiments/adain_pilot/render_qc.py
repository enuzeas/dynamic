"""Movement Style batch9 — 생성 BVH 9개 시각 품질 점검용 막대인형 영상 (content 주인별 1개, 총 3개).

연구 판정용이 아니라 "생성 결과가 제자리 뛰기 content를 눈으로 보기에 유지하는가"를 보는 품질 점검이다.

한 영상 = 4칸 × 2줄
  칸: 실제 제자리 뛰기 / same-person(같은 사람 걷기 스타일) / 다른 사람 스타일 1 / 다른 사람 스타일 2
  줄: 정면(X–Y) / 측면(Z–Y)   — CMU 축: Y 위, +Z 사람 앞, +X 사람 왼쪽
  모든 칸·줄이 같은 축 범위(전체 클립에서 계산)를 쓴다. 클립마다 골반의 평균 수평 위치만 0으로 옮긴다.

입력 (읽기만 함):
  생성: batch9 폴더의 generated.bvh → style_compare.BVH.load → bvh_points
  실제: run_style_pilot.load → to_cmu (생성 결과와 같은 CMU 31 경로, 파일로 저장하지 않음)
출력: batch9 폴더/visual_qc/content_P00X.mp4 (이미 있으면 덮어쓰지 않고 멈춘다)

실행:
  NUMPY_EXPERIMENTAL_DTYPE_API=1 /opt/miniconda3/envs/samsam_pilot/bin/python -B ~/Documents/dynamic_data/scripts/render_qc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_style_pilot as X  # noqa: E402  (NUMPY_EXPERIMENTAL_DTYPE_API=1 검사도 여기서 한다)
import reference_metrics as RM  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
import numpy as np  # noqa: E402

S = X.S
BATCH = RM.OUT
OUT = BATCH / "visual_qc"
PEOPLE = ("P001", "P002", "P003")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
LEFT, RIGHT, SPINE = "#2F6CD3", "#D2582F", "#4E5A57"
BONES = [(a, b, LEFT) for a, b in (("pelvis", "Lhip"), ("Lhip", "Lknee"), ("Lknee", "Lank"), ("Lank", "Ltoe"),
                                   ("neck", "Lsh"), ("Lsh", "Lel"), ("Lel", "Lwr"))] + \
        [(a, b, RIGHT) for a, b in (("pelvis", "Rhip"), ("Rhip", "Rknee"), ("Rknee", "Rank"), ("Rank", "Rtoe"),
                                    ("neck", "Rsh"), ("Rsh", "Rel"), ("Rel", "Rwr"))] + \
        [("pelvis", "neck", SPINE), ("neck", "head", SPINE)]
VIEWS = [("front (X-Y)", 0), ("side (Z-Y)", 2)]  # 가로축으로 쓸 좌표 인덱스, 세로축은 Y(1)


def generated(style: str, content: str) -> dict:
    anim, names, _ = S.BVH.load(str(BATCH / f"pilot_{style}style_{content}content" / "generated.bvh"))
    return S.bvh_points(anim, names)


def real_jog(person: str) -> dict:
    pts, root, _ = X.load(person, "Jogging", RM.RANGES[person]["Jogging"])
    return X.to_cmu(pts, root)[1]


def center(pts: dict) -> dict:
    off = pts["pelvis"].mean(0) * np.array([1.0, 0.0, 1.0])
    return {k: v - off for k, v in pts.items()}


def render(content: str) -> Path:
    others = [p for p in PEOPLE if p != content]
    clips = [(f"Real jog ({content})", real_jog(content)),
             (f"Same: {content}->{content}", generated(content, content))] + \
            [(f"Cross: {s}->{content}", generated(s, content)) for s in others]
    clips = [(t, center(p)) for t, p in clips]
    n = {len(p["pelvis"]) for _, p in clips}
    if len(n) != 1:
        raise SystemExit(f"{content}: 프레임 수가 다름 {n}")
    n = n.pop()
    allp = np.concatenate([np.stack([p[k] for k in p], 1).reshape(-1, 3) for _, p in clips])
    lo, hi = allp.min(0), allp.max(0)
    half = (hi - lo).max() / 2 * 1.08
    mid = (hi + lo) / 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), dpi=80)
    lines = []
    for r, (vname, h) in enumerate(VIEWS):
        for c, (title, _) in enumerate(clips):
            ax = axes[r, c]
            ax.set_xlim(mid[h] - half, mid[h] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color="#BBBBBB", lw=1)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(vname, fontsize=11)
            lines.append([ax.plot([], [], color=col, lw=2.2, solid_capstyle="round")[0] for _, _, col in BONES])
    label = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    dst = OUT / f"content_{content}.mp4"
    writer = FFMpegWriter(fps=60000 / 1001, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(dst), dpi=80):
        for f in range(n):
            i = 0
            for _, h in VIEWS:
                for _, p in clips:
                    for ln, (a, b, _) in zip(lines[i], BONES):
                        ln.set_data([p[a][f, h], p[b][f, h]], [p[a][f, 1], p[b][f, 1]])
                    i += 1
            label.set_text(f"content {content} | absolute AdaIN batch9 | frame {f + 1}/{n} (59.94 fps)")
            writer.grab_frame()
    plt.close(fig)
    return dst


def main() -> None:
    if str(OUT.resolve()).startswith(str(X.WORKTREE.resolve())):
        raise SystemExit(f"저장소 안 경로 금지: {OUT}")
    existing = [p for p in PEOPLE if (OUT / f"content_{p}.mp4").exists()]
    if existing:
        raise SystemExit(f"이미 있음, 덮어쓰지 않음: {existing}")
    OUT.mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        print("저장", render(p))


if __name__ == "__main__":
    main()
