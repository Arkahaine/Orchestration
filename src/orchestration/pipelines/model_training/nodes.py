import torch
from yolov5 import train, detect
import os
import mlflow
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

def evaluate_model(trained_model_path: str) -> dict:
    # Dummy evaluation
    metrics = {"precision": 0.85, "recall": 0.80, "mAP": 0.82}
    mlflow.log_metrics(metrics)
    return metrics

def test_model(trained_model_path: str, test_data_path: str) -> dict:
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    test_images_path = os.path.join(test_data_path, 'images')
    
    results = detect.run(
        weights=trained_model_path,
        source=test_images_path,
        device=device,
        save_txt=True,
        save_conf=True
    )
    
    # Extract numerical metrics from results (dummy example)
    # You need to replace this with actual extraction of numerical values from results
    metrics = {
        "precision": results[0].metrics.precision if results else 0,
        "recall": results[0].metrics.recall if results else 0,
        "mAP": results[0].metrics.mAP if results else 0
    }
    
    mlflow.log_metrics(metrics)
    return metrics