import os
import cv2
import pandas as pd
from pathlib import Path

def preprocess_data(raw_data_path: str) -> pd.DataFrame:
    data = []
    for split in ["train", "valid", "test"]:
        images_path = Path(raw_data_path) / split / "images"
        labels_path = Path(raw_data_path) / split / "labels"
        for image_file in images_path.glob("*.jpg"):
            label_file = labels_path / (image_file.stem + ".txt")
            if label_file.exists():
                data.append({
                    "image_path": str(image_file),
                    "label_path": str(label_file),
                    "split": split
                })
    return pd.DataFrame(data)
