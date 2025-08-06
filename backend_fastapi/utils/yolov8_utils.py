import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server 폴더 기준
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

def load_realesrgan(scale=4):
    model_path = os.path.join(WEIGHTS_DIR, f"RealESRGAN_x{scale}.pth")
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
        half=False
    )
    return upsampler

def load_yolo_model():
    model_path = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    return YOLO(model_path)

upsampler_x4 = load_realesrgan(4)
yolo_model = load_yolo_model()
