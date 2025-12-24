from flask import Flask, request, jsonify
import os
 
app = Flask(__name__)

MODEL_DIR = "/home/site/models/whisper_dysarthria_final"
MODEL_ZIP = "/home/site/models/whisper_dysarthria_final.zip"
MODEL_SAS_URL = os.environ.get("MODEL_SAS_URL")

transcriber = None

# -----------------------------
# LIGHT HOME ROUTE (SAFE)
# -----------------------------
@app.route("/")
def home():
    return "Flask ASR server running ✅"

# -----------------------------
# DOWNLOAD + LOAD (LAZY)
# -----------------------------
def load_model():
    global transcriber
    if transcriber is not None:
        return

    print("Starting model load...")

    import torch
    from transformers import pipeline
    import requests, zipfile

    os.makedirs("/home/site/models", exist_ok=True)

    if not os.path.exists(MODEL_DIR):
        print("Downloading model from Blob Storage...")

        r = requests.get(MODEL_SAS_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall("/home/site/models")

        print("Model downloaded and extracted")

    print("Loading Whisper model...")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_DIR,
        device=-1
    )
    print("Model loaded")

# -----------------------------
# UPLOAD ROUTE
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_audio():
    load_model()

    if "file" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["file"]
    path = "/tmp/audio.wav"
    audio.save(path)

    result = transcriber(path)
    return jsonify({"text": result["text"]})

