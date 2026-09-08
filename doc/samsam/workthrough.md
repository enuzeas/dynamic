# 쌤쌤 뷰어 — 캐릭터 교체·화면 저장 기능 워크스루 (2026-09-08)

두 기능을 `samsam_viewer.html`, `style_transfer.html`에 추가했다. 코드가 실제로 하는 일과, 구현 중 실제로 걸렸던 버그 위주로 정리한다.

## 1. 캐릭터 모델 교체 (.fbx 업로드)

기존엔 두 뷰어 모두 `viewer_data/xbot.fbx`(Mixamo X Bot)를 페이지 로드 시 한 번만 읽어 고정 템플릿으로 썼다. 이제 상단 "🧍 캐릭터 모델 교체 (.fbx)" 버튼으로 다른 FBX를 올리면 그 캐릭터로 3패널을 다시 그린다.

**흐름**: `<input type="file">` → `file.arrayBuffer()` → `new FBXLoader().parse(buf, '')` → SkinnedMesh 존재 확인 → 같은 리타겟 rig(mixamorig 본 이름 매핑)로 패널 재구성.

- `samsam_viewer.html:250` `buildAllPanels(template)` — `computeRig()`로 rig를 먼저 계산(검증 겸)한 뒤에야 기존 렌더러 `dispose()`·캔버스 제거를 하고, 3패널(`SOURCES`)을 새 템플릿으로 다시 세운다.
- `style_transfer.html:248` `buildPanels(template)` — 같은 역할이지만 콘텐츠/스타일/결과 3패널 + 현재 선택된 모션(`loadContent`/`loadStyle`)과 마지막 전이 결과(`lastResult`)를 새 캐릭터에 복원한다.

**리타겟 전제**: 새로 올리는 FBX도 Mixamo 표준 본 이름(`mixamorigHips` 등)을 그대로 써야 한다 — `computeRig()`가 그 이름으로 본을 찾는다.

**실제로 걸린 버그 3개** (Playwright + 시스템 Chrome으로 실행해서 발견):

1. **텍스처 404 → 캐릭터가 새까맣게 렌더됨.** `FBXLoader().parse(buf, '')`는 `path=''`라 FBX가 참조하는 외부 텍스처(.fbm 폴더 등)를 못 찾아 404가 난다. three.js는 텍스처가 아직 로드 안 된 동안 해당 슬롯을 검은 텍스처로 초기화하므로, `material.map`이 그대로 붙어있으면 배경과 구분 안 되는 검은 실루엣이 된다. `stripUnresolvableTextures(obj)`(`samsam_viewer.html:236`, `style_transfer.html:171`)로 업로드 직후 `.map`을 제거해 최소한 `material.color` 기본색으로는 보이게 했다. (브라우저 파일 입력 하나로는 sibling 텍스처 파일에 접근할 방법이 없어 이게 사실상 최선.)
2. **오래된 페이지에서 교체하면 캐릭터가 화면 밖으로 밀려남.** `samsam_viewer.html`은 새 패널의 초기 포즈를 `masterClock.getElapsedTime()`(페이지 로드 후 실제 경과 시간)에 맞춰 재생했다. 카메라 프레이밍은 T포즈 바운딩박스 기준(t=0 가정)이라, 페이지를 열어둔 지 몇 초만 지나도 실제 포즈(t=elapsed)와 카메라 위치가 크게 어긋나 캐릭터가 프레임 밖으로 나갔다. `mixer.update(0)`(`samsam_viewer.html:212`)으로 항상 t=0에서 시작하도록 고쳐 해결.
3. **본 이름이 mixamorig 형식이 아닌 FBX를 올리면 알아보기 힘든 에러로 죽음.** 처음엔 next.md에 "빈 kmByBone으로 조용히 T포즈 멈춤"이라고 추측해 적었는데, 실제로 재현해보니(`mixamorig`→`customrig` 바이너리 문자열 치환으로 가짜 캐릭터 생성) `computeRig()`의 다리 길이 스케일 계산이 `bones['mixamorigHips']`를 무조건 참조하고 있어 `Cannot read properties of undefined (reading 'getWorldScale')`로 그냥 죽었다 — "무경고"가 아니라 "메시지가 불친절한 크래시"였다. `computeRig()`(`samsam_viewer.html:117`, `style_transfer.html:117`) 맨 앞에서 `mixamorigHips` 존재를 확인해 없으면 명확한 메시지로 즉시 throw하도록 고쳤고, `buildAllPanels`/`buildPanels`도 이 검증을 기존 패널을 건드리기 **전에** 하도록 순서를 바꿔 실패해도 화면이 안 비고 이전 캐릭터가 그대로 남는다.

**확인 방법**: 로컬 서버(`python style_transfer_server.py`, :8940) 띄우고 Playwright로 시스템 Chrome을 headless 구동.
- 정상 케이스: `#modelUpload`에 `viewer_data/xbot.fbx`를 그대로 재업로드 → 스크린샷으로 3패널 모두 캐릭터가 다시 렌더되고 애니메이션이 이어지는지 확인. 로컬에 서로 다른 Mixamo 캐릭터 파일이 하나뿐이라(같은 파일 재업로드), 실제 "다른 모양의 캐릭터"로는 아직 검증 못 함(`next.md` 1번).
- 실패 케이스: `xbot.fbx` 바이너리에서 ASCII 문자열 `mixamorig`(9글자)를 길이가 같은 `customrig`(9글자)로 전부 치환한 가짜 FBX를 만들어(오프셋이 안 밀리니 바이너리 FBX 구조가 안 깨짐) 업로드 → 상태 텍스트에 명확한 에러가 뜨고, 스크린샷으로 3패널 모두 교체 전 캐릭터가 그대로 애니메이션 중인지 확인.

## 2. 화면 저장 — 이미지(PNG)/영상(WebM)

두 뷰어 모두 상단에 "📷 이미지 저장"과 "🔴 영상 저장 시작/⏹ 저장 중지" 버튼을 추가했다. 3개 패널의 canvas를 라벨과 함께 오프스크린 캔버스 하나로 합성해 내려받는다.

- `buildComposite()`(`samsam_viewer.html:302`, `style_transfer.html:418`) — `#panels .panel`을 순회해 각 패널의 `<h2>` 제목+`.tag` 텍스트를 라벨로, 그 아래 해당 canvas를 `drawImage`로 이어붙인다.
- **이미지 저장**: `comp.canvas.toBlob(..., 'image/png')` → `downloadBlob()`으로 `<a download>` 트리거.
- **영상 저장**: `comp.canvas.captureStream(30)` + `MediaRecorder`(vp9 우선, 미지원 시 webm 기본)로 녹화. 버튼을 다시 누르면 `recorder.stop()` → `onstop`에서 청크를 모아 `.webm`으로 내려받는다. 녹화 중엔 별도 `requestAnimationFrame` 루프가 합성 캔버스를 매 프레임 다시 그린다.

**실제로 걸린 버그**: 처음 구현 땐 이미지가 라벨만 있고 캐릭터 부분이 통째로 검게 나왔다. 원인은 `THREE.WebGLRenderer`가 기본값 `preserveDrawingBuffer: false`라서, 브라우저가 그 프레임을 화면에 합성한 직후 드로잉 버퍼를 지워버리기 때문 — 저장 버튼 클릭(별도 매크로태스크)이 실행되는 시점엔 이미 지워진 버퍼를 읽게 된다. 두 파일의 렌더러 생성부에 `preserveDrawingBuffer: true`를 추가해 해결(`samsam_viewer.html:191`, `style_transfer.html:194`).

**확인 방법**: Playwright(`acceptDownloads: true`)로 `#saveImgBtn`/`#saveVidBtn` 클릭 → `page.waitForEvent('download')`로 실제 다운로드 발생 확인. PNG는 파일을 열어 3패널 라벨+캐릭터가 다 보이는지 육안 확인(수정 전 24KB짜리 새까만 파일 → 수정 후 136KB, 캐릭터 보임). WebM은 `ffprobe`로 vp9 코덱·2초 길이 확인 후 `ffmpeg`로 20프레임째를 뽑아 캐릭터가 실제로 움직인 자세인지 확인.

**저장 파일명에 라벨 추가**: 처음엔 `samsam_${Date.now()}.png`처럼 타임스탬프뿐이라 여러 장 쌓이면 어떤 캐릭터·스타일 조합이었는지 구분이 안 됐다. `currentModelLabel`(캐릭터 교체 시 파일명에서 갱신)과 `slugFilename()`(파일 시스템에 못 쓰는 문자만 제거, 한글은 유지)을 추가해 `saveFilename(ext)`로 통일:
- `samsam_viewer.html`: `samsam_{캐릭터}_{timestamp}.ext` (예: `samsam_X_Bot_1788836167351.png`)
- `style_transfer.html`: `style_transfer_{캐릭터}_{콘텐츠}_{스타일}_{timestamp}.ext` (예: `style_transfer_X_Bot_07_11_55_07_1788836170169.png`)

Playwright로 기본 상태·캐릭터 교체 후 각각 이미지/영상 저장을 눌러 실제 다운로드 파일명(`download.suggestedFilename()`)에 라벨이 반영되는지 확인.

---

*관련: `samsam_viewer.html`, `style_transfer.html`, `style_transfer_server.py`(로컬 실행용 서버). 세션별 실행 로그는 `samsam_plan.md` 참고.*
