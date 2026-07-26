"""쌤쌤 스타일 전이 UI — 로컬 백엔드.

브라우저에서 콘텐츠 모션(BVH) + 레퍼런스 스타일 모션(BVH)을 고르면 Motion Puzzle을
conda env(motion_puzzle)에서 돌려 스타일 전이 결과 BVH를 만들고, 그걸 X Bot 캐릭터로
보여준다. 스타일 전이 추론은 브라우저에서 못 돌아서(파이토치·conda 필요) 이 백엔드가 필요.
영상→모션(BVH)은 GVHMR(Colab) 단계라 여기선 그 결과·레퍼런스 BVH를 입력으로 받는다.

실행:  python style_transfer_server.py   (→ http://localhost:8940)

정적 파일은 프로젝트 루트에서 그대로 서빙(viewer_data/xbot.fbx, external/motion_puzzle/... BVH가
전부 루트 아래라 별도 라우트 불필요). /api/sources·/api/transfer만 커스텀.
"""
import json
import os
import re
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
MP_DIR = os.path.join(ROOT, "external", "motion_puzzle")
BVH_DIR = os.path.join(MP_DIR, "datasets", "cmu", "test_bvh")     # CMU 샘플 풀
UPLOAD_DIR = os.path.join(MP_DIR, "datasets", "cmu", "uploads")   # 사용자 업로드(촬영→GVHMR 산출물)
OUT_DIR = os.path.join(MP_DIR, "output", "st_ui")
BVH_URL_BASE = "/external/motion_puzzle/datasets/cmu/test_bvh"
UPLOAD_URL_BASE = "/external/motion_puzzle/datasets/cmu/uploads"
OUT_URL_BASE = "/external/motion_puzzle/output/st_ui"

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.bvh$")  # 경로탈출·인젝션 차단(로컬 툴이라도 지킴)
MAX_UPLOAD = 50 * 1024 * 1024  # BVH는 보통 1~수MB, 여유 상한


def list_bvh(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".bvh"))


def resolve_bvh(name):
    """샘플 풀 → 업로드 순으로 이름을 실제 경로로 해석. 없거나 이름이 부적합하면 None."""
    if not name or not NAME_RE.match(name):
        return None
    for d in (BVH_DIR, UPLOAD_DIR):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def run_transfer(content, style):
    """Motion Puzzle 실행 → 스타일 전이된 (footskate 보정) BVH의 URL 반환. 실패 시 (None, err)."""
    cpath, spath = resolve_bvh(content), resolve_bvh(style)
    if not (cpath and spath):
        return None, "잘못된 파일 이름"
    os.makedirs(OUT_DIR, exist_ok=True)
    cstem, sstem = content[:-4], style[:-4]
    out_name = f"Style_{sstem}_Content_{cstem}_fixed.bvh"
    out_path = os.path.join(OUT_DIR, out_name)
    cmd = [
        "conda", "run", "-n", "motion_puzzle", "python", "test.py",
        "--content", cpath, "--style", spath, "--output_dir", OUT_DIR,
    ]
    try:
        r = subprocess.run(cmd, cwd=MP_DIR, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "Motion Puzzle 실행 시간 초과(300초)"
    if r.returncode != 0 or not os.path.isfile(out_path):
        return None, (r.stderr or r.stdout or "알 수 없는 오류")[-1500:]
    return f"{OUT_URL_BASE}/{out_name}", None


def save_upload(name, body):
    """업로드 BVH 저장 → (url, None) 또는 (None, err). CMU 형식만 통과."""
    if not name or not NAME_RE.match(name):
        return None, "파일 이름은 영문·숫자·._- 와 .bvh만 허용됩니다"
    try:
        text = body.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return None, "텍스트 BVH가 아닙니다"
    head = text[:4000]
    if "HIERARCHY" not in head or "ROOT Hips" not in head or "MOTION" not in text:
        return None, "CMU 형식 BVH가 아닙니다 (retarget_smpl_to_cmu.py 출력 또는 CMU BVH만 지원)"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(os.path.join(UPLOAD_DIR, name), "w", encoding="utf-8") as f:
        f.write(text)
    return f"{UPLOAD_URL_BASE}/{name}", None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.path = "/style_transfer.html"
            return super().do_GET()
        if path == "/api/sources":
            sources = [{"name": f, "url": f"{BVH_URL_BASE}/{f}"} for f in list_bvh(BVH_DIR)]
            uploads = [{"name": f, "url": f"{UPLOAD_URL_BASE}/{f}"} for f in list_bvh(UPLOAD_DIR)]
            return self._json(200, {"sources": sources, "uploads": uploads})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        if path == "/api/transfer":
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            url, err = run_transfer(data.get("content", ""), data.get("style", ""))
            return self._json(500 if err else 200, {"error": err} if err else {"url": url})
        if path == "/api/upload":
            if length > MAX_UPLOAD:
                return self._json(413, {"error": f"파일이 너무 큽니다 (>{MAX_UPLOAD // (1024*1024)}MB)"})
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            name = os.path.basename(name)  # 경로 성분 제거
            url, err = save_upload(name, self.rfile.read(length))
            return self._json(400 if err else 200, {"error": err} if err else {"name": name, "url": url})
        return self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # 조용히
        pass


if __name__ == "__main__":
    port = 8940
    print(f"쌤쌤 스타일 전이 UI → http://localhost:{port}  (Ctrl+C로 종료)")
    print(f"입력 BVH 풀: {BVH_DIR} ({len(list_bvh(BVH_DIR))}개), 업로드: {len(list_bvh(UPLOAD_DIR))}개")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
