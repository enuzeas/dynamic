"""원본 촬영본 → 테이크 클립 자동 컷 (사용자가 손으로 자른 data/raw를 정답 예시로 삼는다).

motion_segments.py(화면 전체 픽셀 변화량)는 진행자 이동·손짓에도 반응해 마커가 엉뚱하게
찍혔다. 여기선 사람 관절(MediaPipe) 기반으로 보고, 손으로 자른 클립의 timecode(= Resolve
타임라인 위치 = 원본 내 위치)를 정답으로 써서 규칙을 맞춘다.

작업 순서 (참가자 한 명씩, 끝날 때마다 사람이 확인 — HITL):
  1. 프록시: ffmpeg -i <원본> -map 0:v:0 -map 0:a:0 -vf fps=15,scale=640:360 ... data/proxy/<원본>_proxy.mp4
  2. 관절:   .venv/bin/python auto_cut.py pose data/proxy/<원본>_proxy.mp4
  3. 음성:   mlx_whisper <wav> --model mlx-community/whisper-large-v3-turbo --language ko --word-timestamps True
               --condition-on-previous-text False --hallucination-silence-threshold 2 (→ data/proxy/<원본>.json)
  4. 제안·검토·내보내기 (파이썬에서):
       rows = propose(pose_npz, whisper_json); save_csv(rows, REVIEW_DIR/f"{pid}_proposal.csv")
       review_sheet(pid, src, rows)            # auto/{pid}_timeline.png, _thumbs.png
       (사람이 CSV를 고치거나 keep=N) → export(pid, src, load_csv(...))  # data/raw_auto/ + _nosound/
  2026-10-01 P009–P014를 이 방식으로 자름(P015는 촬영 안 함). 검증: P008 손 컷과 대부분 ±0.5초.

하위 명령 (repo 루트에서 .venv/bin/python):
  pose  <proxy.mp4>                      프록시(15fps 640x360) → <proxy>_pose.npz (33관절 x,y,vis)
  truth <P008> <원본길이목록...>          data/raw 클립 timecode → 원본 기준 정답 구간 출력
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "models" / "pose_landmarker_lite.task"


def df2sec(tc: str) -> float:
    """59.94 drop-frame timecode → 초."""
    h, m, s, f = map(int, re.split("[:;]", tc))
    tm = 60 * h + m
    return (((h * 3600 + m * 60 + s) * 60 + f) - 4 * (tm - tm // 10)) * 1001 / 60000


def probe(path: Path) -> tuple[str, float]:
    o = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream_tags=timecode:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True).stdout)
    tc = next(s["tags"]["timecode"] for s in o["streams"] if "timecode" in s.get("tags", {}))
    return tc, float(o["format"]["duration"])


def pose(proxy: Path) -> Path:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=1))
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                int(i * 1000 / fps))
        out.append([(p.x, p.y, p.visibility) for p in r.pose_landmarks[0]] if r.pose_landmarks else np.full((33, 3), np.nan))
        i += 1
    cap.release()
    lm.close()
    dst = proxy.with_name(proxy.stem + "_pose.npz")
    np.savez(dst, pose=np.array(out, float), fps=fps)
    return dst


def truth(pid: str, timeline: list[tuple[str, float]]) -> list[dict]:
    """timeline: [(원본 파일명, 길이초), ...] — Resolve 타임라인에 01:00:00;00부터 순서대로 놓인 원본들.
    반환: data/raw/{pid}_*.mov 각각의 (원본 파일, 원본 내 시작·끝 초)."""
    starts = np.cumsum([0] + [d for _, d in timeline])
    rows = []
    for clip in sorted((REPO / "data" / "raw").glob(f"{pid}_S1_*_CAM_F.mov")):
        tc, dur = probe(clip)
        t = df2sec(tc) - 3600
        k = int(np.searchsorted(starts, t, side="right") - 1)
        m = re.search(r"_(M0\d)_+T(\d\d)_", clip.name)
        rows.append({"clip": clip.name, "motion": m.group(1), "take": int(m.group(2)), "source": timeline[k][0],
                     "start": round(t - starts[k], 3), "end": round(t - starts[k] + dur, 3)})
    return rows


# ---------------------------------------------------------------------------
# 제안(propose): 진행자 음성(whisper)으로 동작 블록을 찾고, 관절 신호로 테이크 경계를 잡는다.
# 규칙은 P008(사용자가 손으로 자른 정답)에 맞춘 것 — 다른 참가자에선 HITL로 고친다.
# ponytail: 규칙 기반 ±1–2초 정확도. 사람마다 진행 멘트가 많이 다르면 키워드 표를 늘릴 것.
# ---------------------------------------------------------------------------
KEYS = {  # 동작 블록 시작을 알리는 진행자 멘트 (순서대로 찾음)
    "M00": r"T자|티자|T ?포즈|티 ?포즈|양옆",
    "M01": r"걸어|왕복|끝부터|끝까지|왔다 ?갔다",
    "M02": r"벽|짚|뻗",
    "M03": r"검은색|까만|가운데|점으로|점에 ?서|점 있는",
    "M04": r"의자.*앉|앉으실",
    "M05": r"제자리|뛰기",
    "M06": r"조깅|바퀴",
    "M07": r"천천히",
}


def _signals(npz: Path) -> dict:
    from scipy.signal import savgol_filter
    d = np.load(npz)
    P, fps = d["pose"], float(d["fps"])
    P = np.where(np.isnan(P), np.nan, P)
    # 검출 실패 프레임은 직전 값으로 채움
    for j in range(33):
        for c in range(3):
            a = P[:, j, c]
            idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
            np.maximum.accumulate(idx, out=idx)
            P[:, j, c] = a[idx]
    P = np.nan_to_num(P)
    sm = lambda a, w=9: savgol_filter(a, w, 2)
    hip = (P[:, 23, :2] + P[:, 24, :2]) / 2
    sh = (P[:, 11, :2] + P[:, 12, :2]) / 2
    L = float(np.median(np.linalg.norm(sh - hip, axis=1)))
    hy = sm(hip[:, 1])
    jump = hip[:, 1] - sm(hip[:, 1], 15)
    return {
        "fps": fps, "t": np.arange(len(P)) / fps, "L": L,
        "hx": sm(hip[:, 0], 15),
        "hyn": (hy - np.median(hy)) / L,  # 앉으면 +0.5 근처
        "speed": np.convolve(np.r_[0, np.linalg.norm(np.diff(hip, axis=0), axis=1)] * fps / L, np.ones(9) / 9, "same"),
        "tpose": (np.abs(P[:, 15, 0] - P[:, 16, 0]) / L > 1.2)
                 & ((np.abs(P[:, 15, 1] - P[:, 11, 1]) + np.abs(P[:, 16, 1] - P[:, 12, 1])) / 2 / L < 0.3),
        "jump": np.convolve(jump ** 2, np.ones(15) / 15, "same") / L ** 2,
    }


def _segs(whisper_json: Path) -> list[tuple[float, float, str]]:
    d = json.loads(whisper_json.read_text())
    return [(s["start"], s["end"], s["text"].strip()) for s in d["segments"] if s["text"].strip()]


def _words(whisper_json: Path) -> list[tuple[float, float, str]]:
    d = json.loads(whisper_json.read_text())
    return [(w["start"], w["end"], w["word"].strip()) for s in d["segments"] for w in s.get("words", []) if w["word"].strip()]


def _find(words, pat, after, before=1e9):
    """단어 3개 창으로 정규식을 찾고, 일치가 시작되는 단어의 (시작, 끝, 창 텍스트)를 돌려준다."""
    for i, w in enumerate(words):
        if not (after <= w[0] < before):
            continue
        win = " ".join(x[2] for x in words[i:i + 3])
        m = re.search(pat, win)
        if m and m.start() < len(w[2]):
            return (w[0], w[1], win)
    return None


def _runs(mask: np.ndarray, t: np.ndarray, min_len: float = 0.0) -> list[tuple[float, float]]:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(t[a], t[min(b, len(t) - 1)]) for a, b in zip(e[::2], e[1::2]) if t[min(b, len(t) - 1)] - t[a] >= min_len]


def _crossings(x, t, lo, hi, level, sign=None):
    i = np.flatnonzero((t[:-1] >= lo) & (t[:-1] < hi) & (np.sign(x[:-1] - level) != np.sign(x[1:] - level)))
    out = [(t[k], np.sign(x[k + 1] - x[k])) for k in i]
    return [c for c in out if sign is None or c[1] == sign]


def propose(npz: Path, whisper_json: Path) -> list[dict]:
    S, segs, words = _signals(npz), _segs(whisper_json), _words(whisper_json)
    t, end_t = S["t"], S["t"][-1]
    a = {}
    prev = 0.0
    for m, pat in KEYS.items():
        s = _find(words, pat, prev)
        a[m] = s
        prev = s[0] if s else prev
    nxt = lambda m: next((a[k][0] for k in list(KEYS)[list(KEYS).index(m) + 1:] if a[k]), end_t)
    out = []
    add = lambda m, s, e, note="": out.append({"motion": m, "start": round(float(s), 2), "end": round(float(e), 2), "note": note})

    # M00: T포즈 시작 0.8초 전 ~ 걷기 시작 직전
    if a["M00"]:
        tp = _runs(S["tpose"] & (t >= a["M00"][0] - 3) & (t < nxt("M00")), t, 1.0)
        s0 = tp[0][0] - 0.8 if tp else a["M00"][1]
        move = t[(t > (a["M01"][0] if a["M01"] else s0 + 10)) & (S["speed"] > 0.3)]
        instr = max((g[0] for g in segs if a["M01"] and g[0] <= a["M01"][0]), default=None)  # 걷기 안내 문장 시작
        e0 = min(move[0] - 0.3 if len(move) else s0 + 15.5, (instr + 2.0) if instr is not None else 1e9)
        add("M00", s0, e0, "" if tp else "T포즈 미검출")

    # M01: 같은 쪽 끝(방향 전환점) 사이 = 왕복 1회
    if a["M01"]:
        from scipy.signal import find_peaks
        lo, hi = a["M01"][0], nxt("M01")
        i0, i1 = np.searchsorted(t, [lo, hi])
        x = S["hx"][i0:i1]
        ex = sorted([(t[i0 + i], "R") for i in find_peaks(x, prominence=0.15)[0]] +
                    [(t[i0 + i], "L") for i in find_peaks(-x, prominence=0.15)[0]])
        span = x.max() - x.min() if len(x) else 0
        ex = [e for e in ex if (e[1] == "L" and S["hx"][np.searchsorted(t, e[0])] < x.min() + 0.2 * span)
              or (e[1] == "R" and S["hx"][np.searchsorted(t, e[0])] > x.max() - 0.2 * span)]  # 끝까지 안 간 전환은 무시
        same = [e[0] for e in ex if e[1] == ex[0][1]] if ex else []
        for s, e in zip(same, same[1:]):
            add("M01", s, e)

    # M02: "내리시고" 다음 첫 지시("대세요/뻗어/짚…")마다 새 테이크. "한 번/두 번"처럼 세는 말은
    # 직전 지시와 1.2초 넘게 떨어져 있으면 내리라는 말이 없어도 새 테이크로 본다.
    # 거리 조정 멘트("뒤로/한 발/가깝")가 나오면 직전 지시는 버리고 다시 받는다.
    # (관절 기반 뻗기 감지도 시도했지만 옆으로 선 사람은 먼 팔이 가려져 신호가 뒤집혀서 음성 기준을 유지)
    if a["M02"]:
        lo, hi = a["M02"][0], nxt("M02")
        cmds, armed = [], True
        blk = [w for w in words if lo <= w[0] < hi]
        for k, w in enumerate(blk):
            win = " ".join(x[2] for x in blk[k:k + 2])
            count = re.match(r"(한|두|세|네|다섯) ?번(?! ?더)", win)  # "한번 더 해볼게요"는 안내라 제외
            if re.search(r"뒤로|앞으로|발자국|가깝", w[2]) and cmds and not armed:
                cmds.pop()
                armed = True
                continue
            if (armed and (re.search(r"대세요|대시고|댈|대볼|댔|대는|붙|뻗어|뻗었|짚|올렸|올려", w[2]) or count)) \
                    or (count and cmds and w[0] - cmds[-1] > 1.2):
                cmds.append(w[0] + 0.2)
                armed = False
            if re.search(r"내리|내렸|내려", w[2]):
                armed = True
        for s, e in zip(cmds, cmds[1:] + [hi + 1.6]):
            add("M02", s, e)

    # M03: 첫 질문이 끝난 뒤 ~ 다음 동작 안내 직전, 질문 시작점 기준으로 4등분
    if a["M03"]:
        lo, hi = a["M03"][0], nxt("M03") - 0.4
        qs = [s for s in segs if lo <= s[0] < hi and "?" in s[2]]
        first = next((g for g in segs if lo + 0.5 < g[0] < hi and re.search(r"\?|까요|주세요|해주", g[2])), None)  # 안내 다음 첫 질문
        s0 = first[1] if first else lo
        cand = [q[0] for q in qs[1:]]
        cuts = [s0]
        for k in (1, 2, 3):
            target = s0 + (hi - s0) * k / 4
            near = min(cand, key=lambda c: abs(c - target)) if cand else target
            cuts.append(near if abs(near - target) < 4 else target)  # 질문이 멀면 그냥 등분점
        cuts = sorted(set(cuts)) + [hi]
        for s, e in zip(cuts, cuts[1:]):
            add("M03", s, e, "4등분(질문 시작점에 맞춤)")

    # M04: 앉은 구간마다 — 일어서기 3.5초 전(앉은 뒤 0.5초 이후) ~ 일어선 뒤 1.4초
    if a["M04"]:
        lo, hi = a["M04"][0], nxt("M04")
        for s, e in _runs((S["hyn"] > 0.3) & (t >= lo) & (t < hi), t, 0.5):
            add("M04", max(s + 0.8, e - 3.7), e + 1.4)

    # M05: 위아래 진동이 큰 구간
    if a["M05"]:
        lo, hi = a["M05"][0], nxt("M05")
        r = _runs((S["jump"] > 3e-4) & (t >= lo) & (t < hi), t, 1.0)  # 조용히 서 있을 때 상위 5%가 ~1e-4
        bouts = []  # 3초 넘게 쉬면 다른 세트로 본다
        for s, e in r:
            if bouts and s - bouts[-1][1] < 3.0:
                bouts[-1] = (bouts[-1][0], e)
            else:
                bouts.append((s, e))
        for s, e in bouts:
            if e - s >= 4.0:
                add("M05", s + 0.2, e - 0.4)

    # M06/M07: 화면 가운데를 같은 방향으로 지날 때마다 한 바퀴
    for m in ("M06", "M07"):
        if not a[m]:
            continue
        lo, hi = a[m][0], nxt(m)
        if m == "M06":
            go = _find(words, r"시작", lo, hi)
            lo = go[0] if go else lo
        else:  # M07: 감속 구간은 빼고, "한 바퀴만 더/크게 한 바퀴" 안내 뒤부터
            go = _find(words, r"바퀴", lo, hi)
            lo = go[0] if go else lo
        # 출발 = "시작" 뒤 처음 움직인 순간, 이후 출발 지점(x)을 같은 방향으로 다시 지날 때마다 한 바퀴
        mv = np.flatnonzero((t >= lo) & (t < hi) & (S["speed"] > 0.8))
        if not len(mv):
            continue
        k0 = mv[0]
        level = S["hx"][k0]
        k1 = min(len(t) - 1, k0 + int(S["fps"]))
        sign = np.sign(S["hx"][k1] - level)
        laps = [t[k0]]
        for c, sg in _crossings(S["hx"], t, t[k0] + 2.0, hi, level):
            if sg == sign and c - laps[-1] > 3.0:
                laps.append(c)
        if m == "M06":
            m06_level = level
        elif len(laps) < 2 and "m06_level" in dir():
            # 이미 걷고 있는 중이라 출발점이 애매할 때: 조깅 때 기준선을 안내 3초 전부터 같은 방향으로 두 번 지나는 구간
            cr = _crossings(S["hx"], t, lo - 3.0, hi, m06_level)
            nxt_same = next((c for c, sg in cr[1:] if sg == cr[0][1]), None) if cr else None
            laps = [cr[0][0], nxt_same] if nxt_same else laps
        for s, e in zip(laps, laps[1:]):
            add(m, s, e)

    for m in KEYS:  # 테이크 번호
        for k, r in enumerate([r for r in out if r["motion"] == m], 1):
            r["take"] = k
    missing = [m for m in KEYS if not a[m]]
    if missing:
        out.append({"motion": "-", "start": 0, "end": 0, "take": 0, "note": "멘트 못 찾음: " + ",".join(missing)})
    return out


def score(prop: list[dict], truth_rows: list[dict]) -> None:
    """정답(손 컷)과 비교: 같은 동작·테이크 번호끼리 시작/끝 오차."""
    for r in truth_rows:
        p = next((p for p in prop if p["motion"] == r["motion"] and p.get("take") == r["take"]), None)
        if p:
            print(f"{r['motion']} T{r['take']:02d}  정답 {r['start']:7.2f}–{r['end']:7.2f}  제안 {p['start']:7.2f}–{p['end']:7.2f}"
                  f"  오차 {p['start'] - r['start']:+5.2f} / {p['end'] - r['end']:+5.2f}")
        else:
            print(f"{r['motion']} T{r['take']:02d}  정답 {r['start']:7.2f}–{r['end']:7.2f}  제안 없음")
    extra = [p for p in prop if not any(p["motion"] == r["motion"] and p.get("take") == r["take"] for r in truth_rows)]
    for p in extra:
        print(f"(정답에 없는 제안) {p['motion']} T{p.get('take', 0):02d} {p['start']:.2f}–{p['end']:.2f} {p['note']}")


# ---------------------------------------------------------------------------
# 검토(HITL)용 자료 + 내보내기
# ---------------------------------------------------------------------------
SRC_DIR = Path("/Volumes/LUMIX/촬영Data")
# Resolve 타임라인(01:00:00;00부터) 원본 배치 순서 — data/raw의 timecode가 이 기준
# 참가자: 58→P008 59→P009 60→P010 61→P011 63→P012 64→P013 65→P014, 62(24초, 촬영 안 함)→P015
# (participants.csv 키 기록 기준. 촬영 순서대로면 62가 P012여야 하지만 키가 안 맞아 이쪽을 택함)
TIMELINE = ["P1026258.MOV", "P1026259.MOV", "P1026260.MOV", "P1026261.MOV",
            "P1026262.MOV", "P1026263.MOV", "P1026264.MOV", "P1026265.MOV"]
REVIEW_DIR = REPO / "data" / "metadata" / "segment_review" / "auto"
OUT_DIR = REPO / "data" / "raw_auto"


def save_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion", "take", "start", "end", "dur", "keep", "note"])
        for r in rows:
            if r["motion"] != "-":
                w.writerow([r["motion"], r["take"], f"{r['start']:.2f}", f"{r['end']:.2f}",
                            f"{r['end'] - r['start']:.1f}", "Y", r.get("note", "")])


def load_csv(path: Path) -> list[dict]:
    import csv
    return [{"motion": r["motion"], "take": int(r["take"]), "start": float(r["start"]), "end": float(r["end"])}
            for r in csv.DictReader(path.open()) if r["keep"].strip().upper() == "Y"]


def review_sheet(pid: str, src: str, rows: list[dict], truth_rows: list[dict] | None = None) -> Path:
    """신호 타임라인 + 진행자 멘트 + 제안 구간(+정답 있으면 함께)을 한 장으로, 테이크별 썸네일 한 장."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    stem = Path(src).stem
    S = _signals(REPO / "data" / "proxy" / f"{stem}_proxy_pose.npz")
    segs = _segs(REPO / "data" / "proxy" / f"{stem}.json")
    t = S["t"]
    cols = dict(zip(KEYS, ["gray", "C0", "C1", "C2", "C3", "C4", "C5", "C6"]))
    n_rows = int(np.ceil(t[-1] / 120))
    fig, axes = plt.subplots(n_rows, 1, figsize=(26, 4.2 * n_rows), squeeze=False)
    for k, ax in enumerate(axes[:, 0]):
        lo, hi = k * 120, (k + 1) * 120
        ax.plot(t, S["hx"], lw=0.8, label="좌우 위치")
        ax.plot(t, 0.5 + 0.3 * np.clip(S["hyn"], -1, 1.5), lw=0.8, label="앉음(골반 높이)")
        ax.plot(t, np.clip(S["speed"], 0, 3) / 3, lw=0.6, alpha=0.6, label="이동 속도")
        for r in rows:
            if r["motion"] in cols and r["end"] > lo and r["start"] < hi:
                ax.axvspan(r["start"], r["end"], ymin=0.55, ymax=1, color=cols[r["motion"]], alpha=0.35)
                ax.text((r["start"] + r["end"]) / 2, 1.08, f"{r['motion']}\nT{r['take']:02d}", ha="center", fontsize=8)
                ax.axvline(r["start"], color="k", lw=0.6)
        for r in truth_rows or []:
            if r["end"] > lo and r["start"] < hi:
                ax.axvspan(r["start"], r["end"], ymin=0, ymax=0.08, color="k", alpha=0.5)
        for s in segs:
            if lo <= s[0] < hi:
                ax.text(s[0], -0.32, s[2][:18], fontsize=6, rotation=30, ha="left", va="top", color="dimgray")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xticks(np.arange(lo, hi + 1, 5))
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{pid} ({src}) 자동 컷 제안 — 위 색띠 = 제안 테이크" + (", 아래 검은띠 = 손으로 자른 정답" if truth_rows else ""))
    plt.tight_layout()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tl = REVIEW_DIR / f"{pid}_timeline.png"
    plt.savefig(tl, dpi=70)
    plt.close()

    # 테이크별 첫·중간·끝 프레임
    cap = cv2.VideoCapture(str(REPO / "data" / "proxy" / f"{stem}_proxy.mp4"))
    keep = [r for r in rows if r["motion"] in cols]
    fig, axes = plt.subplots(len(keep), 3, figsize=(9, 1.9 * len(keep)), squeeze=False)
    for i, r in enumerate(keep):
        for j, tt in enumerate([r["start"] + 0.2, (r["start"] + r["end"]) / 2, r["end"] - 0.2]):
            cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
            ok, fr = cap.read()
            ax = axes[i, j]
            if ok:
                ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{r['motion']} T{r['take']:02d}  {['시작', '중간', '끝'][j]} {tt:.1f}s", fontsize=7)
    cap.release()
    plt.tight_layout()
    th = REVIEW_DIR / f"{pid}_thumbs.png"
    plt.savefig(th, dpi=80)
    plt.close()
    return tl


def sec2df(sec: float) -> str:
    """초 → 59.94 drop-frame timecode (df2sec의 역)."""
    fn = int(round(sec * 60000 / 1001))
    d, m = divmod(fn, 35964)  # 10분당 프레임 수
    fn += 36 * d + (4 * ((m - 4) // 3596) if m > 3 else 0)
    f = fn % 60
    s = fn // 60 % 60
    mi = fn // 3600 % 60
    h = fn // 216000
    return f"{h:02d}:{mi:02d}:{s:02d};{f:02d}"


def export(pid: str, src: str, rows: list[dict], only: set[str] | None = None) -> list[Path]:
    """원본에서 테이크를 잘라 data/raw_auto/ 와 data/raw_auto/_nosound/ 에 data/raw와 같은 형식으로 저장.
    timecode는 Resolve 타임라인 기준(01:00:00;00 + 원본 배치 위치 + 컷 시작)이라 truth()로 다시 읽힌다."""
    offset = sum(probe(SRC_DIR / s)[1] for s in TIMELINE[:TIMELINE.index(src)])
    (OUT_DIR / "_nosound").mkdir(parents=True, exist_ok=True)
    done = []
    for r in rows:
        name = f"{pid}_S1_{r['motion']}_T{r['take']:02d}_CAM_F.mov"
        if only and name not in only:
            continue
        out, ns = OUT_DIR / name, OUT_DIR / "_nosound" / name
        tc = sec2df(3600 + offset + r["start"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['start']:.3f}", "-i", str(SRC_DIR / src),
                        "-t", f"{r['end'] - r['start']:.3f}",
                        "-filter_complex", "[0:a:0][0:a:1]amerge=inputs=2,pan=stereo|c0=c0+c1|c1=c0+c1[a]",
                        "-map", "0:v:0", "-map", "[a]",
                        "-vf", "format=yuv420p", "-color_range", "tv",
                        "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
                        "-c:v", "libx264", "-profile:v", "high", "-crf", "15", "-preset", "fast",
                        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-timecode", tc, str(out)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-map", "0:v:0", "-c", "copy", "-an",
                        "-timecode", tc, str(ns)], check=True)
        done.append(out)
    return done


def demo() -> None:
    for tc in ["01:00:00;00", "01:00:20;39", "01:05:40;58", "01:38:34;39", "01:10:00;00", "01:01:00;04", "01:00:59;59"]:
        assert sec2df(df2sec(tc)) == tc, (tc, sec2df(df2sec(tc)))
    print("demo ok")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pose":
        print(pose(Path(sys.argv[2])))
    elif cmd == "truth":
        tl = [(Path(p).name, probe(Path(p))[1]) for p in sys.argv[3:]]
        for r in truth(sys.argv[2], tl):
            print(json.dumps(r, ensure_ascii=False))
