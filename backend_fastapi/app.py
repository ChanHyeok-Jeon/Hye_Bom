from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import cv2
import numpy as np
import io
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from ultralytics import YOLO
import os
import traceback

app = FastAPI()

WEIGHTS_DIR = "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_realesrgan(scale=4):
    model_path = os.path.join(WEIGHTS_DIR, f"RealESRGAN_x{scale}.pth")
    # RRDBNet 생성자 시그니처에 맞게 키워드 인자로 명확히 전달
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=scale,
        num_feat=64,
        num_block=23,
        num_grow_ch=32
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        dni_weight=None,
        device=device,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    return upsampler

def load_yolo_model():
    model_path = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    return YOLO(model_path)

upsampler_x4 = load_realesrgan(4)
yolo_model = load_yolo_model()

@app.get("/")
async def root():
    return {"message": "FastAPI 서버 정상 작동 중입니다."}

@app.post("/process")
async def process_image(file: UploadFile = File(...), scale: int = 4):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        upscaled_img, _ = upsampler_x4.enhance(img_bgr, outscale=scale)

        results = yolo_model(upscaled_img)
        annotated_img = results[0].plot()

        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)})
