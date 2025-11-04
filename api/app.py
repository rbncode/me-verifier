import os
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import joblib
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

load_dotenv()

app = Flask(__name__)

MODEL_VERSION = "me-verifier-v1"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

device = None
mtcnn = None
resnet = None
model = None
scaler = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    global device, mtcnn, resnet, model, scaler

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Flask API running on device: {device}")

    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
    model_file = os.path.join(MODEL_PATH, 'model.joblib')
    scaler_file = os.path.join(MODEL_PATH, 'scaler.joblib')

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError("Model or scaler not found. Please train the model first.")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

def preprocess_image(image_bytes):
    try:
        img = Image.open(image_bytes).convert('RGB')

        face_tensor = mtcnn(img)

        if face_tensor is None:
            return None, "No face detected in the image."

        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor)

        return embedding.cpu().numpy().flatten(), None

    except Exception as e:
        return None, f"Error during image processing: {e}"


@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/verify', methods=['POST'])
def verify_image():
    start_time = time.time()
    THRESHOLD = float(os.getenv('THRESHOLD', 0.75))
    MAX_MB = float(os.getenv('MAX_MB', 10))

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    if file_size_mb > MAX_MB:
        return jsonify({"error": f"File size exceeds the limit of {MAX_MB} MB"}), 413
    file.seek(0)

    embedding, error = preprocess_image(file.stream)
    if error:
        return jsonify({"error": error}), 400

    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    prediction = model.predict(embedding_scaled)[0]
    score = model.predict_proba(embedding_scaled)[0][1] # Probability of class 1 ("me")

    is_me = bool(prediction == 1 and score >= THRESHOLD)

    timing_ms = (time.time() - start_time) * 1000

    response = {
        "model_version": MODEL_VERSION,
        "is_me": is_me,
        "score": round(float(score), 4),
        "threshold": THRESHOLD,
        "timing_ms": round(timing_ms, 2)
    }

    return jsonify(response), 200

if __name__ == '__main__':
    try:
        load_models()
        app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))
    except FileNotFoundError as e:
        print(f"Startup error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
