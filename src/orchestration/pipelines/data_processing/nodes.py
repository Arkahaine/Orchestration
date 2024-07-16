import os
import cv2
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(raw_data_path: str) -> pd.DataFrame:
    """
    Preprocess raw data by collecting image and label file paths for training, validation, and testing splits.

    Args:
    - raw_data_path (str): Path to the raw data directory.

    Returns:
    - pd.DataFrame: DataFrame containing paths to images and corresponding label files along with the data split type.
    """
    data = []
    supported_image_formats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    for split in ["train", "valid", "test"]:
        images_path = Path(raw_data_path) / split / "images"
        labels_path = Path(raw_data_path) / split / "labels"

        if not images_path.exists() or not labels_path.exists():
            logging.warning(f"Missing directory: {images_path} or {labels_path}")
            continue

        for img_format in supported_image_formats:
            for image_file in images_path.glob(img_format):
                label_file = labels_path / (image_file.stem + ".txt")
                if label_file.exists():
                    data.append({
                        "image_path": str(image_file),
                        "label_path": str(label_file),
                        "split": split
                    })
                else:
                    logging.warning(f"Label file {label_file} for image {image_file} not found.")

    if not data:
        logging.error("No data found. Please check the raw data directory and file formats.")

    return pd.DataFrame(data)

def validate_data(data_df: pd.DataFrame):
    """
    Validate the data by checking the existence of image and label files.

    Args:
    - data_df (pd.DataFrame): DataFrame containing paths to images and corresponding label files.
    """
    missing_images = data_df[~data_df['image_path'].apply(lambda x: Path(x).exists())]
    missing_labels = data_df[~data_df['label_path'].apply(lambda x: Path(x).exists())]

    if not missing_images.empty:
        logging.warning(f"Missing images:\n{missing_images['image_path'].tolist()}")
    if not missing_labels.empty:
        logging.warning(f"Missing labels:\n{missing_labels['label_path'].tolist()}")

def load_and_display_image(image_path: str):
    """
    Load an image using OpenCV and display it.

    Args:
    - image_path (str): Path to the image file.
    """
    image = cv2.imread(image_path)
    if image is not None:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logging.error(f"Failed to load image: {image_path}")

def main(raw_data_path: str):
    """
    Main function to preprocess data, validate it, and display a sample image.

    Args:
    - raw_data_path (str): Path to the raw data directory.
    """
    data_df = preprocess_data(raw_data_path)
    if not data_df.empty:
        validate_data(data_df)
        sample_image_path = data_df.iloc[0]['image_path']
        load_and_display_image(sample_image_path)
    else:
        logging.error("Data preprocessing failed. No data to process.")

if __name__ == "__main__":
    raw_data_path = "C:/Users/arkah/Desktop/orchestration/data/01_raw"  # Replace with your actual raw data path
    main(raw_data_path)
