import torch
from yolov5 import detect
import os
import mlflow

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
