from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import io
from PIL import Image
import numpy as np
import cv2
import traceback
from utils.yolov8_utils import yolo_model, upsampler_x4

router = APIRouter()

@router.post("/process")
async def process_image(request: Request, file: UploadFile = File(...), scale: int = 4):
    client_host = request.client.host  # 클라이언트 IP 얻기
    print(f"접속 IP: {client_host}")  # 서버 콘솔에 출력 (또는 파일, DB에 저장 가능)

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        upscaled_img, _ = upsampler_x4.enhance(img_bgr, outscale=scale)
        upscaled_img_contiguous = np.ascontiguousarray(upscaled_img)  # 추가

        results = yolo_model(upscaled_img_contiguous)

        annotated_img = results[0].plot()

        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
