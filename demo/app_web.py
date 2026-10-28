# app_web.py
# 업로드 → 업스케일 → YOLOv8(best.pt) → 건강 요약(모의/실제)까지 한 화면에 표시하는 FastAPI 단일 파일 데모
# 실행: uvicorn app_web:app --reload
# 필요: pip install fastapi uvicorn ultralytics opencv-python openai python-multipart
# 주의: 본 정보는 참고용이며 이상/악화 시 의료전문가 상담 권장

import os
import base64
import io
import uuid
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from openai import OpenAI

# ================== 설정 ==================
# YOLO 가중치 경로
WEIGHTS_PATH = os.environ.get("TONGUE_WEIGHTS", "best.pt")

# (선택) 실제 GPT 호출을 쓰고 싶다면 아래 플래그를 True로, 그리고 API 키를 넣으세요.
USE_REAL_GPT_DEFAULT = False  # 기본은 모의 응답
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()  # 여기에 하드코딩도 가능: "sk-or-..."
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo").strip()
# =========================================

app = FastAPI(title="혀봄 데모", version="1.0")

# 결과물 저장 폴더(선택) - 이번 예제는 base64 인라인으로 바로 보여주므로 필수는 아님
os.makedirs("outputs", exist_ok=True)

# ------------- 유틸 -------------
def np_to_datauri(img_bgr: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    """OpenCV ndarray(BGR)를 data:image/...;base64 로 변환"""
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    encode_param = []
    if ext in [".jpg", ".jpeg"]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(ext, img_bgr, encode_param)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.replace(".", "")
    return f"data:image/{mime};base64,{b64}"

def filebytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def upscale_bicubic(img_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return up

def run_yolo_array(weights_path: str, img_bgr: np.ndarray, imgsz: int = 640, conf: float = 0.25):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {weights_path}")
    model = YOLO(weights_path)
    results = model.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    best_idx = confs.argmax()
    best_conf = float(confs[best_idx])
    best_cls_id = int(cls_ids[best_idx])
    class_name = r.names.get(best_cls_id, str(best_cls_id))
    annotated_bgr = r.plot()  # BGR
    return best_cls_id, class_name, best_conf, annotated_bgr, r.names

def build_client() -> Optional[OpenAI]:
    api_key = OPENROUTER_API_KEY
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

# ----- 클래스별 모의 응답 템플릿 (앱 UI용 5줄 불릿 고정 형식) -----
MOCK_SUMMARIES: Dict[str, str] = {
    "담홍설": """• 상태 요약: 옅은 붉은빛의 정상 범주에 가까운 혀 상태입니다.
• 가능 증상: 경미한 피로감·체력 저하가 간헐적으로 동반될 수 있습니다.
• 생활관리: 수분 충분 섭취, 규칙적인 수면(7–8시간)
• 병원 기준: 현기증·심한 창백감·심계 항진이 지속되면 진료 권장
• 약국 조치: 종합비타민 또는 비타민B군 보충 고려""",
    "홍설": """• 상태 요약: 혀가 붉고 열감 징후가 나타나는 상태입니다.
• 가능 증상: 구강건조, 갈증 증가, 입마름, 열감
• 생활관리: 맵고 짠 음식·알코올 줄이기, 미지근한 물 자주 마시기
• 병원 기준: 지속적인 구강 통증·궤양·발열이 동반되면 진료
• 약국 조치: 구강건조용 보습제, 비타민C·아연 보충 고려""",
    "담백설": """• 상태 요약: 혀 색이 희고 창백해 보이는 상태입니다.
• 가능 증상: 피로·무기력, 어지럼, 소화불량이 동반될 수 있음
• 생활관리: 철분·단백질 충분 섭취(살코기·콩류), 규칙적 수면
• 병원 기준: 어지럼·두근거림·심한 창백감 지속 시 내과 진료
• 약국 조치: 철분 보충제·종합비타민 등 일반 영양보조제 고려""",
    "열문설": """• 상태 요약: 혀 표면에 균열(열문)이 보이는 상태입니다.
• 가능 증상: 구강건조, 따가움, 매운 음식 섭취 시 통증
• 생활관리: 수분 섭취 증가, 자극적 음식·과음 피하기
• 병원 기준: 궤양·출혈·통증 지속 시 치과/구강내과 진료
• 약국 조치: 구강 보습 젤, 비타민B군 보충 고려""",
    "박태": """• 상태 요약: 얇고 옅은 설태가 낀 상태입니다.
• 가능 증상: 가벼운 소화불편 또는 일시적 피로감
• 생활관리: 식사 규칙 지키기, 과식·야식 줄이기
• 병원 기준: 설태 변색·구취 심화 지속 시 진료
• 약국 조치: 유산균·소화제 등 단기적 보조 고려""",
    "후태": """• 상태 요약: 설태가 두껍게 쌓인 상태입니다.
• 가능 증상: 소화저하, 구취, 혀의 답답한 느낌
• 생활관리: 기름진 음식 줄이고, 식후 양치·혀클리너 가볍게 사용
• 병원 기준: 발열·오심·복통 동반 또는 설태 색이 진하게 지속되면 진료
• 약국 조치: 소화제, 프로바이오틱스 등 보조 고려""",
    "황태": """• 상태 요약: 노란빛 설태가 관찰되는 상태입니다.
• 가능 증상: 구강건조, 구취, 위열/소화불량 징후
• 생활관리: 기름지고 매운 음식 피하기, 수분 섭취 증가
• 병원 기준: 복통·설사·발열 동반 시 소화기 내과 진료
• 약국 조치: 제산제·소화제, 구강헹굼제 단기 사용 고려""",
}
DEFAULT_MOCK = """• 상태 요약: 혀 상태 변화가 관찰되었습니다.
• 가능 증상: 가벼운 구강 불편감 또는 소화 증상
• 생활관리: 수분 섭취·균형 잡힌 식사·규칙적 수면
• 병원 기준: 통증·궤양·출혈 등 이상 지속 시 진료
• 약국 조치: 증상에 맞는 일반의약품 단기 사용 고려"""

def get_mock_summary(tongue_type: str) -> str:
    return MOCK_SUMMARIES.get(tongue_type, DEFAULT_MOCK)

def get_gpt_summary(tongue_type: str) -> str:
    client = build_client()
    if client is None:
        return get_mock_summary(tongue_type)

    prompt = f"""
당신은 한의학 및 일반 건강정보에 정통한 건강 어드바이저입니다.
사용자의 혀 상태가 '{tongue_type}'(으)로 감지되었습니다.

요구사항:
- 앱에 표시될 "매우 간단한 요약"으로 작성(불릿 5줄 이내)
- 포함: (1) 상태 한줄 요약 (2) 관련 가능 증상/주의 신호 (3) 생활관리 2가지 (4) 병원 진료 필요 기준 (5) 약국에서 시도 가능한 일반적 조치(가능 시 예시)
- 한국 사용자 기준 용어 사용, 과장 금지, 확정적 진단 금지
- 위험징후 시 즉시 진료 권고 포함
- 길게 쓰지 말고 핵심만, 문장은 간결하게

출력형식(그대로):
• 상태 요약: ...
• 가능 증상: ...
• 생활관리: ...
• 병원 기준: ...
• 약국 조치: ...
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "의료 정보는 참고용이며, 진단/치료는 의료기관과 상담해야 합니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content.strip()
        if not text or "•" not in text:
            return get_mock_summary(tongue_type)
        return text
    except Exception:
        return get_mock_summary(tongue_type)

# ------------- 뷰 -------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>혀봄 데모</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 12px; }}
    .card {{ border: 1px solid #bbb3; border-radius: 14px; padding: 16px; box-shadow: 0 1px 8px #00000010; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 16px; }}
    .col {{ display: flex; flex-direction: column; gap: 8px; }}
    .imgbox {{ border: 1px dashed #8886; border-radius: 10px; padding: 8px; display: flex; justify-content: center; align-items: center; min-height: 260px; background: #f7f7f7; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    .meta {{ font-size: 0.92rem; color: #888; }}
    .summary {{ white-space: pre-line; font-size: 1.0rem; line-height: 1.5; }}
    .footer {{ margin-top: 10px; font-size: 0.9rem; color: #777; }}
    .row {{ display: flex; gap: 8px; align-items: center; }}
    .btn {{ padding: 10px 14px; border-radius: 10px; border: 1px solid #ccc7; background: #fff; cursor: pointer; }}
    .btn:hover {{ background: #f4f4f4; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>👅 혀봄 데모</h1>
  <p class="meta">이미지 업로드 → 업스케일링 → YOLO 판독 → 건강 요약(채팅 스타일)까지 한 번에 보여줍니다.</p>
  <form class="card" action="/analyze" method="post" enctype="multipart/form-data">
    <div class="row" style="flex-wrap: wrap;">
      <input type="file" name="file" accept="image/*" required class="btn"/>
      <input class="btn" type="submit" value="분석 시작"/>
    </div>
    <div class="meta">* 가중치: {weights_path} / 입력 이미지는 서버에서만 처리되고 저장하지 않습니다.</div>
  </form>
  {result_block}
  <div class="footer">※ 본 정보는 참고용입니다. 증상이 지속·악화되면 의료전문가와 상담하세요.</div>
</body>
</html>
"""

RESULT_BLOCK = """
<section class="card" style="margin-top: 16px;">
  <div class="grid">
    <div class="col">
      <h3>원본</h3>
      <div class="imgbox">{img_orig}</div>
    </div>
    <div class="col">
      <h3>업스케일</h3>
      <div class="imgbox">{img_up}</div>
    </div>
    <div class="col">
      <h3>YOLO 판독(주석)</h3>
      <div class="imgbox">{img_anno}</div>
      <div class="meta">클래스: <b>{cls_name}</b> (id={cls_id}) · 신뢰도: {conf:.3f}</div>
      <div class="meta">모델 클래스 목록: {class_list}</div>
    </div>
  </div>
</section>

<section class="card" style="margin-top: 16px;">
  <h3>건강 요약</h3>
  <div class="summary">{summary}</div>
</section>
"""


# ------------- 라우트 -------------
@app.get("/", response_class=HTMLResponse)
def index():
    html = INDEX_HTML.format(
        weights_path=WEIGHTS_PATH,
        result_block=""
    )
    return HTMLResponse(html)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    # 파일 읽기
    raw = await file.read()
    orig = filebytes_to_bgr(raw)
    if orig is None:
        return HTMLResponse("이미지를 읽을 수 없습니다.", status_code=400)

    # 업스케일
    up = upscale_bicubic(orig, scale=2)

    # YOLO 추론
    try:
        yolo_out = run_yolo_array(WEIGHTS_PATH, up, imgsz=640, conf=0.25)
    except FileNotFoundError as e:
        return HTMLResponse(f"<pre>{e}</pre>", status_code=500)

    if yolo_out is None:
        cls_id = -1
        cls_name = "검출 없음"
        conf = 0.0
        anno = up.copy()
        class_list = "N/A"
        summary_text = "• 상태 요약: 검출 결과가 없습니다.\n• 가능 증상: -\n• 생활관리: 촬영 시 혀 전체가 밝고 선명하게 나오도록 재촬영 권장\n• 병원 기준: 통증·궤양 등 이상 시 진료\n• 약국 조치: -"
    else:
        cls_id, cls_name, conf, anno, names = yolo_out
        class_list = ", ".join(list(names.values()))
        # 항상 모의 요약만 사용
        summary_text = get_mock_summary(cls_name)

    # 이미지들을 data URI로 인라인
    img_orig_uri = np_to_datauri(orig, ".jpg")
    img_up_uri = np_to_datauri(up, ".jpg")
    img_anno_uri = np_to_datauri(anno, ".jpg")

    result_html = RESULT_BLOCK.format(
        img_orig=f'<img src="{img_orig_uri}"/>',
        img_up=f'<img src="{img_up_uri}"/>',
        img_anno=f'<img src="{img_anno_uri}"/>',
        cls_name=cls_name,
        cls_id=cls_id,
        conf=conf,
        class_list=class_list,
        summary=summary_text
    )

    page_html = INDEX_HTML.format(
        weights_path=WEIGHTS_PATH,
        result_block=result_html
    )
    return HTMLResponse(page_html)

