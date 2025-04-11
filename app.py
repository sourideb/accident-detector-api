"""
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    img = Image.open(io.BytesIO(image.read()))

    # Dummy logic to simulate accident detection
    # You can replace this with a real ML model later
    print("Received an image for analysis.")

    # Just pretending: if image size is big, assume 'accident'
    if img.size[0] > 500:
        result = 'Accident Detected'
        # You can add code here to trigger alerts
    else:
        result = 'No Accident'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""





from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import io

# Initialize app
app = Flask(__name__)

# Load model and processor
processor = DetrImageProcessor.from_pretrained("hilmantm/detr-traffic-accident-detection")
model = DetrForObjectDetection.from_pretrained("hilmantm/detr-traffic-accident-detection")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")

    # Prepare and infer
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # Prepare output
    detections = []
    for score, label in zip(results["scores"], results["labels"]):
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 2)
        detections.append({"label": label_name, "confidence": confidence})

    return jsonify({"predictions": detections})

@app.route('/')
def index():
    return "Accident Detection API is running!"

if __name__ == '__main__':
    app.run(debug=True)
