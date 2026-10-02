"""FootMR(GVHMR) 일괄 처리 — Colab에서 footmr_colab.ipynb 설치(셀 1-9)를 마친 뒤 실행한다. 2026-10-03 77개 클립에 사용.

  !cd /content/FootMR && nohup python3.10 /content/gvhmr_colab_batch.py > /content/batch.log 2>&1 &

- 드라이브 samsam_gvhmr/inputs의 클립을 order.txt 순서로, ViT-H 모델은 한 번만 올려서 처리한다.
- 59.94fps → 29.97fps로 2프레임 중 1개만 남긴다: GVHMR은 30fps로 학습됐고 demo.py도 30fps로 간주한다.
- 렌더링(incam/global 영상)은 건너뛴다: 각도 분석에는 hmr4d_results.pt만 필요하다.
- 결과는 드라이브 outputs/<클립>/에 바로 저장. 이미 있으면 건너뛴다(중단 후 재실행 가능).
- 주의: 이 셀을 colab-mcp로 띄우면 그 뒤 셀 실행 요청이 응답 없이 시간 초과된다. 진행은 드라이브 동기화 폴더로 본다.
다음 단계는 로컬 gvhmr_to_npz.py.
"""
import importlib.util, os, shutil, subprocess, sys, time
from pathlib import Path

os.chdir('/content/FootMR'); sys.path.insert(0, '/content/FootMR')
import torch, hydra
G = Path('/content/drive/MyDrive/samsam_gvhmr')
OUT = G / 'outputs'; OUT.mkdir(exist_ok=True)
TMP = Path('/content/in30'); TMP.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location('demo', 'tools/demo.py')
demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)

_cache = {}
def reuse(name):  # 큰 ViT-H 두 개는 클립마다 다시 올리지 않는다 (YOLO 추적기는 매번 새로)
    orig = getattr(demo, name)
    def get(*a, **k):
        if name not in _cache:
            _cache[name] = orig(*a, **k)
        return _cache[name]
    setattr(demo, name, get)
for name in ('VitPoseExtractor', 'Extractor'):
    reuse(name)

order = [l.strip() for l in (G / 'order.txt').read_text().splitlines() if l.strip()]
model, t0, fails = None, time.time(), {}
while True:
    todo = [c for c in order if not (OUT / Path(c).stem / 'hmr4d_results.pt').exists() and fails.get(c, 0) < 2]
    ready = [c for c in todo if (G / 'inputs' / c).exists()]
    if not todo: break
    if not ready: print('동기화 대기...', len(todo), flush=True); time.sleep(60); continue
    c = ready[0]; stem = Path(c).stem; tic = time.time()
    try:
        src = TMP / f'{stem}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(G / 'inputs' / c), '-vf', 'fps=30000/1001',
                        '-c:v', 'libx264', '-crf', '17', '-an', str(src)], check=True)
        sys.argv = ['demo', '--video', str(src), '-s', '--no_postproc']
        cfg = demo.parse_args_to_cfg()
        demo.run_preprocess(cfg)
        data = demo.load_data_dict(cfg)
        if model is None:
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
        with torch.no_grad():
            pred = demo.detach_to_cpu(model.predict(data, static_cam=True, no_postproc=True))
        dst = OUT / stem; dst.mkdir(exist_ok=True)
        torch.save(pred, dst / 'hmr4d_results.pt.tmp'); os.replace(dst / 'hmr4d_results.pt.tmp', dst / 'hmr4d_results.pt')
        shutil.copy(cfg.paths.vitpose, dst / 'vitpose.pt')
        shutil.rmtree(cfg.output_dir, ignore_errors=True); src.unlink()
        done = len(order) - len(todo) + 1
        print(f'[{done}/{len(order)}] {stem} {int(data["length"])}f {time.time()-tic:.0f}s (누적 {(time.time()-t0)/60:.1f}분)', flush=True)
    except Exception as e:
        fails[c] = fails.get(c, 0) + 1
        print(f'실패 {stem}: {type(e).__name__}: {e}', flush=True)
print('BATCH_DONE', {k: v for k, v in fails.items() if v >= 2}, flush=True)
