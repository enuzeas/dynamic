"""캐리커처 미리보기 — 조깅(M06) 개인 스타일을 "그대로 / 과장"으로 나란히 보여주는 막대인형 영상.

토요일 회의의 "캐리커처(과장) vs 사진(그대로)" 결정을 눈으로 하기 위한 자료.
얼굴 캐리커처처럼 "평균에서 벗어난 만큼을 α배"한다 — 단 아무 차이나 키우지 않고,
pca_pilot에서 반복 촬영 간 ICC가 높았던 축(그 사람다운 차이)만 키운다(테이크마다 다른 노이즈는 그대로).

포즈 공간 (pca_pilot과 동일 정규화: 골반 중심, 몸통 길이로 나눔, 코+12관절 2D):
  μ_g = 전체 평균 PC 점수,  μ_p = 그 사람의 평균,  s_p = 그 사람 PC 점수 표준편차 / 전체 표준편차
  z'(t) = μ_g + α·(μ_p − μ_g) + (z(t) − μ_p)·s_p^(α−1)      ← 자세(평균)와 움직임 폭(표준편차)을 각각 α배
  α=0: 평균형(그 사람의 타이밍만 남고 자세·폭은 평균),  α=1: 원본,  α>1: 과장
  "움직임 폭만" 판은 자세 항(μ_p − μ_g)은 α=1로 두고 폭만 키운다 — 체형(팔다리 비율)이 섞인 자세 성분을 빼고 보기 위함.
골반 이동 경로는 원본 그대로 둔다(같은 트랙을 도는 모습).

사용법 (repo 루트, pca_pilot.py를 먼저 한 번 돌려 캐시가 있어야 함):
  .venv/bin/python caricature_preview.py                 # reports/caricature/M06_caricature.mp4 + M06_metrics.md
패널 아래 숫자: 팔 흔들림(손목이 어깨 기준으로 움직이는 폭)·무릎 폭(무릎 위아래 범위)은 몸통 길이 단위,
팔꿈치(굽힘 각, 180°=쭉 폄)·상체 기울기(수직 대비)는 도. 괄호는 12명 원본 평균 대비.
  .venv/bin/python caricature_preview.py --alpha 3 --people P002,P012,P006
  .venv/bin/python caricature_preview.py --web           # 슬라이더 웹페이지 (틀: caricature_lab.html)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_extract import _smooth  # noqa: E402
from build_icc_features import _trim_window  # noqa: E402
from evaluate_icc import icc_3_1  # noqa: E402
import pca_pilot as Q  # noqa: E402

J = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 코 + 어깨·팔꿈치·손목·골반·무릎·발목
IDX = {j: i for i, j in enumerate(J)}
BONES_L = [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27)]
BONES_R = [(12, 14), (14, 16), (12, 24), (24, 26), (26, 28)]
BONES_C = [(11, 12), (23, 24)]
OUT = Path("reports/caricature")


def frames(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """트림 구간의 (정규화 포즈 (T, 26), 골반 경로 (T, 2)) — 둘 다 몸통 길이 단위."""
    d = Q.dyn8(rec)["dyn"]
    win = _trim_window(np.mean([d[j]["speed"] for j in ("오른엉덩이", "왼엉덩이")], axis=0))
    xy = np.stack([_smooth(rec["xy"][:, j].astype(float)) for j in range(33)], axis=1)[win]
    hip = (xy[:, 23] + xy[:, 24]) / 2
    # 프레임별 몸통 길이(약 1초 창으로 평활)로 나눔 — 원을 돌며 카메라에 가까워졌다 멀어지는 원근 변화를 뺀다.
    # (클립 중앙값으로 나누면 가까울 때 인형이 커짐. pca_pilot 식별률은 두 방식이 비슷: 폭 75→79%, 자세+폭 92→88%)
    from scipy.signal import savgol_filter
    Lf = np.linalg.norm((xy[:, 11] + xy[:, 12]) / 2 - hip, axis=1)
    L = savgol_filter(Lf, min(61, len(Lf) // 2 * 2 - 1), 2)[:, None, None]
    return ((xy[:, J] - hip[:, None]) / L).reshape(len(xy), -1), hip / np.median(Lf)


def caricature(z: np.ndarray, mu_g, sd_g, mu_p, sd_p, alpha: float, pose_axes, amp_axes) -> np.ndarray:
    out = z.copy()
    s = np.where(sd_g > 0, sd_p / sd_g, 1.0)
    for c in range(z.shape[1]):
        a_pose = alpha if c in pose_axes else 1.0
        a_amp = alpha if c in amp_axes else 1.0
        out[:, c] = mu_g[c] + a_pose * (mu_p[c] - mu_g[c]) + (z[:, c] - mu_p[c]) * s[c] ** (a_amp - 1)
    return out


def metrics(pose: np.ndarray) -> dict[str, float]:
    """사람이 읽을 수 있는 지표 (몸통 길이 단위·도). pose: (T, 26) 골반 중심 정규화 좌표."""
    P = pose.reshape(len(pose), -1, 2)
    g = lambda j: P[:, IDX[j]]
    def ang(a, b, c):
        u, v = g(a) - g(b), g(c) - g(b)
        return np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1)))
    swing = np.mean([np.sqrt((g(w) - g(sh)).var(0).sum()) for w, sh in ((15, 11), (16, 12))])
    knee = np.mean([np.ptp(np.percentile(g(k)[:, 1], [5, 95])) for k in (25, 26)])
    trunk = (g(11) + g(12)) / 2  # 골반 중심이 원점이라 이게 곧 몸통 벡터
    lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1])).mean()
    return {"팔 흔들림": float(swing), "팔꿈치": float(np.mean([ang(11, 13, 15).mean(), ang(12, 14, 16).mean()])),
            "무릎 폭": float(knee), "상체 기울기": float(lean)}


def label(m: dict, ref: dict) -> str:
    pct = lambda k: f"{(m[k] / ref[k] - 1) * 100:+.0f}%"
    deg = lambda k: f"{m[k] - ref[k]:+.0f}°"
    return (f"팔 흔들림 {m['팔 흔들림']:.2f} ({pct('팔 흔들림')})   무릎 폭 {m['무릎 폭']:.2f} ({pct('무릎 폭')})\n"
            f"팔꿈치 {m['팔꿈치']:.0f}° ({deg('팔꿈치')})   상체 기울기 {m['상체 기울기']:.0f}° ({deg('상체 기울기')})")


def draw(ax, pose: np.ndarray, root: np.ndarray, title: str, lim, text: str = "") -> None:
    p = pose.reshape(-1, 2) + root
    ax.clear()
    for bones, col in ((BONES_L, "#2E6FD8"), (BONES_R, "#D8572E"), (BONES_C, "#555555")):
        for a, b in bones:
            ax.plot(*zip(p[IDX[a]], p[IDX[b]]), color=col, lw=3, solid_capstyle="round")
    neck = (p[IDX[11]] + p[IDX[12]]) / 2
    ax.plot(*zip(neck, p[IDX[0]]), color="#555555", lw=3)
    ax.add_patch(__import__("matplotlib").patches.Circle(p[IDX[0]], 0.28, color="#555555"))
    ax.set_xlim(*lim[0])
    ax.set_ylim(lim[1][1], lim[1][0])  # 영상 좌표: y 아래로
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    if text:
        ax.text(0.5, -0.02, text, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333333")


def export_web(template: Path = Path("caricature_lab.html")) -> Path:
    """caricature_lab.html(틀)에 조깅·앉기 데이터를 넣어 reports/caricature/caricature_lab.html로 저장.
    브라우저에서 슬라이더로 과장 배율을 바꾸면 같은 식(caricature())을 JS로 다시 계산한다."""
    import json
    cache = pickle.loads(Q.CACHE.read_bytes())
    labels = {"M06": "조깅 (M06)", "M04": "앉았다 일어서기 (M04)"}
    keys = {"팔 흔들림": "swing", "무릎 폭": "knee", "팔꿈치": "elbow", "상체 기울기": "lean"}
    out = {"motions": {}}
    for motion, k, parts, _ in Q.RUNS:
        people = parts.split(",")
        full = {(p, t): frames(cache[str(Q.RAW / f"{p}_S1_{motion}_T{t:02d}_CAM_F.mov")]) for p in people for t in range(1, k + 1)}
        data = {key: (f[::2], r[::2]) for key, (f, r) in full.items()}  # 59.94 → 약 30fps
        fps = cache[str(Q.RAW / f"{people[0]}_S1_{motion}_T01_CAM_F.mov")]["fps"] / 2
        allf = np.concatenate([f for f, _ in data.values()])
        mean = allf.mean(0)
        _, _, vt = np.linalg.svd(allf - mean, full_matrices=False)
        Z = {key: (f - mean) @ vt.T for key, (f, _) in data.items()}
        allz = np.concatenate(list(Z.values()))
        tab = lambda fn: np.array([[fn(Z[(p, t)]) for t in range(1, k + 1)] for p in people])
        icc_mean = [float(icc_3_1(tab(lambda z: z[:, c].mean()))) for c in range(10)]
        icc_std = [float(icc_3_1(tab(lambda z: z[:, c].std()))) for c in range(10)]
        allm = {key: {**{keys[m]: v for m, v in metrics(f).items()}, "bounce": float(np.std(r[:, 1]))} for key, (f, r) in full.items()}
        metric_icc = {m: float(icc_3_1(np.array([[allm[(p, t)][m] for t in range(1, k + 1)] for p in people])))
                      for m in ("swing", "knee", "elbow", "lean", "bounce")}
        pts = allf.reshape(-1, 2)
        lo, hi = np.percentile(pts, 0.2, axis=0) - 0.5, np.percentile(pts, 99.8, axis=0) + 0.5
        r3 = lambda a: [round(float(v), 3) for v in np.ravel(a)]
        persons = {}
        for p in people:
            zp = np.concatenate([Z[(p, t)] for t in range(1, k + 1)])
            persons[p] = {"mu": r3(zp.mean(0)), "sd": r3(zp.std(0)),
                          "takes": [{"x": r3(data[(p, t)][0]), "b": r3(data[(p, t)][1][:, 1] - data[(p, t)][1][:, 1].mean())}
                                    for t in range(1, k + 1)]}
        out["motions"][motion] = {
            "label": labels[motion], "k": k, "fps": round(fps, 3), "people": people,
            "mean": r3(mean), "V": [[round(float(v), 5) for v in row] for row in vt],
            "mu_g": r3(allz.mean(0)), "sd_g": r3(allz.std(0)), "icc_mean": icc_mean, "icc_std": icc_std,
            "metric_icc": metric_icc, "view": [r3(lo), r3(hi)], "persons": persons,
            "source": f"데이터: S1 정면 촬영 {len(people)}명 × {k}테이크, MediaPipe 2D 관절, 골반 중심·프레임별 몸통 길이 정규화. "
                      + ("P005·P009는 화면 이탈로 제외." if motion == "M06" else "P010·P011은 이 동작을 촬영하지 않음."),
        }
    html = template.read_text().replace("/*__DATA__*/null", json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "caricature_lab.html"
    dst.write_text(html)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--people", default=None, help="쉼표구분. 기본: 평균에서 가장 먼 2명 + 가장 평균적인 1명")
    ap.add_argument("--icc", type=float, default=0.75, help="이 값 이상인 축만 과장")
    ap.add_argument("--web", action="store_true", help="슬라이더로 과장을 바꾸는 웹페이지(reports/caricature/caricature_lab.html)만 만든다")
    args = ap.parse_args()
    if args.web:
        dst = export_web()
        print("저장:", dst, f"({dst.stat().st_size / 1e6:.1f} MB)")
        return

    motion, k, parts, _ = Q.RUNS[0]
    people = parts.split(",")
    cache = pickle.loads(Q.CACHE.read_bytes())
    data = {(p, t): frames(cache[str(Q.RAW / f"{p}_S1_{motion}_T{t:02d}_CAM_F.mov")]) for p in people for t in range(1, k + 1)}

    allf = np.concatenate([f[::2] for f, _ in data.values()])
    mean = allf.mean(0)
    _, _, vt = np.linalg.svd(allf - mean, full_matrices=False)  # 전체 축(26개) — α=1이면 정확히 원본 복원
    Z = {key: (f - mean) @ vt.T for key, (f, _) in data.items()}
    mu_g = np.concatenate(list(Z.values())).mean(0)
    sd_g = np.concatenate(list(Z.values())).std(0)

    # 축별 ICC — 테이크 간 일정한 축만 과장 대상(상위 10개 축 안에서)
    icc_mean = {c: icc_3_1(np.array([[Z[(p, t)][:, c].mean() for t in range(1, k + 1)] for p in people])) for c in range(10)}
    icc_std = {c: icc_3_1(np.array([[Z[(p, t)][:, c].std() for t in range(1, k + 1)] for p in people])) for c in range(10)}
    pose_axes = {c for c, v in icc_mean.items() if v >= args.icc}
    amp_axes = {c for c, v in icc_std.items() if v >= args.icc}
    print("과장 대상 — 자세 축:", sorted(c + 1 for c in pose_axes), " 움직임 폭 축:", sorted(c + 1 for c in amp_axes))

    per = {p: (np.concatenate([Z[(p, t)] for t in range(1, k + 1)]).mean(0),
               np.concatenate([Z[(p, t)] for t in range(1, k + 1)]).std(0)) for p in people}
    if args.people:
        pick = args.people.split(",")
    else:  # 신뢰 축 기준으로 평균에서 얼마나 떨어져 있나(표준화 거리)
        def dist(p):
            m, s = per[p]
            dm = [(m[c] - mu_g[c]) / sd_g[c] for c in pose_axes]
            ds = [np.log(s[c] / sd_g[c]) for c in amp_axes]
            return float(np.sqrt(np.sum(np.square(dm + ds))))
        order = sorted(people, key=dist)
        pick = [order[-1], order[-2], order[0]]
        print("평균과의 거리:", {p: round(dist(p), 2) for p in order})
    print("영상에 넣는 사람:", pick)

    cols = [("평균형 (×0)", 0.0, "both"), ("원본 (×1)", 1.0, "both"),
            (f"과장 ×{args.alpha:g} — 움직임 폭만", args.alpha, "amp"), (f"과장 ×{args.alpha:g} — 자세+폭", args.alpha, "both")]
    clips = {}
    for p in pick:
        z = Z[(p, 1)]
        root = data[(p, 1)][1]
        root = np.c_[np.zeros(len(root)), root[:, 1] - root[:, 1].mean()]  # 몸을 따라가는 화면: 좌우 이동은 빼고 들썩임만
        mu_p, sd_p = per[p]
        clips[p] = []
        for _, a, mode in cols:
            zz = caricature(z, mu_g, sd_g, mu_p, sd_p, a, pose_axes if mode == "both" else set(), amp_axes)
            clips[p].append(((zz @ vt) + mean, root))
    assert np.allclose(clips[pick[0]][1][0], data[(pick[0], 1)][0], atol=1e-8), "α=1이 원본과 달라짐"

    # 사람 간 비교용 지표: 기준 = 12명 원본 첫 테이크의 평균
    orig = {p: metrics(data[(p, 1)][0]) for p in people}
    ref = {m: float(np.mean([orig[p][m] for p in people])) for m in next(iter(orig.values()))}
    bounce = {p: float(np.std(data[(p, 1)][1][:, 1])) for p in people}
    texts = {p: [label(metrics(pose), ref) for pose, _ in clips[p]] for p in pick}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["| 참가자 | 팔 흔들림 | 무릎 폭 | 팔꿈치 | 상체 기울기 | 위아래 들썩임 | 영상 |", "|---|---|---|---|---|---|---|"]
    for p in sorted(people):
        m = orig[p]
        rows.append(f"| {p} | {m['팔 흔들림']:.2f} ({(m['팔 흔들림'] / ref['팔 흔들림'] - 1) * 100:+.0f}%) | "
                    f"{m['무릎 폭']:.2f} ({(m['무릎 폭'] / ref['무릎 폭'] - 1) * 100:+.0f}%) | {m['팔꿈치']:.0f}° ({m['팔꿈치'] - ref['팔꿈치']:+.0f}°) | "
                    f"{m['상체 기울기']:.0f}° ({m['상체 기울기'] - ref['상체 기울기']:+.0f}°) | {bounce[p]:.2f} | {'✓' if p in pick else ''} |")
    rows.append(f"| **평균** | {ref['팔 흔들림']:.2f} | {ref['무릎 폭']:.2f} | {ref['팔꿈치']:.0f}° | {ref['상체 기울기']:.0f}° | "
                f"{np.mean(list(bounce.values())):.2f} | |")
    # 지표마다 반복 촬영 간 일관성(ICC) — 높을수록 "그 사람다운" 특징
    allm = {key: {**metrics(f), "들썩임": float(np.std(r[:, 1]))} for key, (f, r) in data.items()}
    icc_m = {m: icc_3_1(np.array([[allm[(p, t)][m] for t in range(1, k + 1)] for p in people]))
             for m in ("팔 흔들림", "무릎 폭", "팔꿈치", "상체 기울기", "들썩임")}
    rows.append("| 반복 촬영 ICC | " + " | ".join(f"{icc_m[m]:.2f}" for m in ("팔 흔들림", "무릎 폭", "팔꿈치", "상체 기울기", "들썩임")) + " | |")
    (OUT / f"{motion}_metrics.md").write_text(
        f"# {motion} 사람별 지표 (원본, 첫 테이크)\n\n단위: 몸통 길이(어깨 중점–골반 중점) = 1. 괄호는 {len(people)}명 평균 대비.\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    plt.rcParams["font.family"] = "AppleGothic"
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.06, hspace=0.45, wspace=0.05)
    fig.suptitle(f"조깅(M06) 캐리커처 미리보기 — 반복 촬영에서 일정했던 축(ICC≥{args.icc:g})만 과장 · 2D 막대인형", fontsize=14)
    pts = np.concatenate([(pose.reshape(len(pose), -1, 2) + root[:, None]).reshape(-1, 2)
                          for p in pick for pose, root in clips[p]])
    lo, hi = np.percentile(pts, 0.5, axis=0) - 0.35, np.percentile(pts, 99.5, axis=0) + 0.35
    lim = ((lo[0], hi[0]), (lo[1], hi[1]))
    n = max(len(clips[p][0][0]) for p in pick)
    fps = cache[str(Q.RAW / f"{pick[0]}_S1_{motion}_T01_CAM_F.mov")]["fps"]
    step = 2  # 59.94 → 약 30fps
    dst = OUT / f"{motion}_caricature.mp4"
    writer = FFMpegWriter(fps=fps / step, bitrate=4000)
    with writer.saving(fig, str(dst), dpi=100):
        for i in range(0, n, step):
            for r, p in enumerate(pick):
                for c, (title, _, _) in enumerate(cols):
                    pose, root = clips[p][c]
                    j = i % len(pose)  # 짧은 테이크는 반복
                    draw(axes[r, c], pose[j], root[j], f"{p} · {title}", lim, texts[p][c])
            writer.grab_frame()
    plt.close(fig)
    print("저장:", dst)


if __name__ == "__main__":
    main()
