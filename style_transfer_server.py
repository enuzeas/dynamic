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
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
MP_DIR = os.path.join(ROOT, "external", "motion_puzzle")
BVH_DIR = os.path.join(MP_DIR, "datasets", "cmu", "test_bvh")
OUT_DIR = os.path.join(MP_DIR, "output", "st_ui")
BVH_URL_BASE = "/external/motion_puzzle/datasets/cmu/test_bvh"
OUT_URL_BASE = "/external/motion_puzzle/output/st_ui"

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.bvh$")  # 경로탈출·인젝션 차단(로컬 툴이라도 지킴)


def list_bvh():
    if not os.path.isdir(BVH_DIR):
        return []
    return sorted(f for f in os.listdir(BVH_DIR) if f.endswith(".bvh"))


def valid_name(name):
    return bool(name) and bool(NAME_RE.match(name)) and os.path.isfile(os.path.join(BVH_DIR, name))


def run_transfer(content, style):
    """Motion Puzzle 실행 → 스타일 전이된 (footskate 보정) BVH의 URL 반환. 실패 시 (None, err)."""
    if not (valid_name(content) and valid_name(style)):
        return None, "잘못된 파일 이름"
    os.makedirs(OUT_DIR, exist_ok=True)
    cstem, sstem = content[:-4], style[:-4]
    out_name = f"Style_{sstem}_Content_{cstem}_fixed.bvh"
    out_path = os.path.join(OUT_DIR, out_name)
    cmd = [
        "conda", "run", "-n", "motion_puzzle", "python", "test.py",
        "--content", os.path.join(BVH_DIR, content),
        "--style", os.path.join(BVH_DIR, style),
        "--output_dir", OUT_DIR,
    ]
    try:
        r = subprocess.run(cmd, cwd=MP_DIR, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "Motion Puzzle 실행 시간 초과(300초)"
    if r.returncode != 0 or not os.path.isfile(out_path):
        return None, (r.stderr or r.stdout or "알 수 없는 오류")[-1500:]
    return f"{OUT_URL_BASE}/{out_name}", None


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
            files = list_bvh()
            items = [{"name": f, "url": f"{BVH_URL_BASE}/{f}"} for f in files]
            return self._json(200, {"sources": items})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/transfer":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        url, err = run_transfer(data.get("content", ""), data.get("style", ""))
        if err:
            return self._json(500, {"error": err})
        return self._json(200, {"url": url})

    def log_message(self, fmt, *args):  # 조용히
        pass


if __name__ == "__main__":
    port = 8940
    print(f"쌤쌤 스타일 전이 UI → http://localhost:{port}  (Ctrl+C로 종료)")
    print(f"입력 BVH 풀: {BVH_DIR} ({len(list_bvh())}개)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
