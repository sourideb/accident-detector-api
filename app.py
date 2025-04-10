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
