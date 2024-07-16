import torch
from yolov5 import train
import os
from pathlib import Path

def train_yolo(data_yaml: str, img_size: int, epochs: int, batch_size: int) -> str:
    device = '0' if torch.cuda.is_available() else 'cpu'
    data_yaml_full_path = os.path.join(os.getcwd(), data_yaml)
    
    result = train.run(
        weights='yolov5n.pt',  # Specify YOLOv5n model
        data=data_yaml_full_path,
        epochs=epochs,
        imgsz=img_size,
        batch_size=batch_size,
        device=device  # Pass '0' for the first CUDA device or 'cpu' for CPU
    )
    
    # Assuming the model path is saved in runs/train/exp/weights/best.pt
    latest_run_dir = sorted(Path('runs/train').glob('exp*'), key=os.path.getmtime)[-1]
    model_path = latest_run_dir / 'weights/best.pt'
    
    return str(model_path)
