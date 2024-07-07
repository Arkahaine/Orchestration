from flask import Flask, request, send_file, render_template
from PIL import Image
import io
import torch
import os
import shutil
import matplotlib.pyplot as plt
from yolov5 import detect
from pathlib import Path

app = Flask(__name__)

# Function to get the latest 'best.pt' model
def get_latest_model():
    try:
        latest_run_dir = sorted(Path('runs/train').glob('exp*'), key=os.path.getmtime)[-1]
        model_path = latest_run_dir / 'weights/best.pt'
        return model_path
    except IndexError:
        return None

def draw_predictions(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    
    # Clear the current figure
    plt.clf()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for pred in predictions:
        rect = plt.Rectangle(
            (pred["x_center"] - pred["width"] / 2, pred["y_center"] - pred["height"] / 2),
            pred["width"], pred["height"], fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        label = f'{pred["class"]}'
        if pred["confidence"] is not None:
            label += f': {pred["confidence"]:.2f}'
        ax.text(
            pred["x_center"], pred["y_center"] - pred["height"] / 2, label, 
            bbox=dict(facecolor='yellow', alpha=0.5), clip_box=ax.clipbox, clip_on=True)
    
    plt.axis('off')
    temp_output_path = 'temp_output.jpg'
    plt.savefig(temp_output_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    return temp_output_path

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']

    if file.filename == '':
        return "Empty file", 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return "Invalid image format", 400

    # Save the image temporarily to disk
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # Load the latest model
    model_path = get_latest_model()
    if not model_path or not model_path.exists():
        return "Model not found", 500

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set the directory to save results and clear previous results
    results_dir = 'runs/detect/exp'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    
    # Make predictions and save results
    detect.run(weights=str(model_path), source=temp_image_path, device=device if device == 'cpu' else 0, save_txt=True, project=results_dir, name='results', exist_ok=True)
    
    # Read the results from the txt file
    result_txt_path = Path(results_dir) / 'results' / 'labels' / 'temp_image.txt'
    predictions = []
    if result_txt_path.exists():
        with open(result_txt_path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                predictions.append({
                    "class": int(parts[0]),
                    "x_center": float(parts[1]) * image.width,
                    "y_center": float(parts[2]) * image.height,
                    "width": float(parts[3]) * image.width,
                    "height": float(parts[4]) * image.height,
                    "confidence": float(parts[5]) if len(parts) > 5 else None
                })

    # Draw predictions on the image
    output_image_path = draw_predictions(temp_image_path, predictions)
    
    # Remove the temporary image
    os.remove(temp_image_path)

    return send_file(output_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)