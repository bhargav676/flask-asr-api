from flask import Flask, request, jsonify
import os
import torch
from transformers import pipeline 
import requests 
import zipfile

app = Flask(__name__)

MODEL_DIR = "/home/site/models/whisper_dysarthria_final"
MODEL_ZIP = "/home/site/models/whisper_dysarthria_final.zip"
MODEL_SAS_URL = os.environ.get("MODEL_SAS_URL")

transcriber = None

@app.route("/")
def home():
    return "Flask ASR server running (Whisper-small)"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs("/home/site/models", exist_ok=True)

        print("Downloading model from Blob Storage...")
        r = requests.get(MODEL_SAS_URL)
        with open(MODEL_ZIP, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall("/home/site/models")

        print("Model downloaded and extracted")

def load_model():
    global transcriber
    if transcriber is None:
        download_and_extract_model()
        print("Loading Whisper-small model...")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=MODEL_DIR,
            device=-1  # CPU
        )
        print("Model loaded")

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

if __name__ == "__main__":
    app.run()


