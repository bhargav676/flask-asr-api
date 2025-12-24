from flask import Flask, request, Response
import os
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading fine-tuned model...")
model_path = "whisper_dysarthria_final"

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    device=-1   
)

print("Model loaded")

@app.route("/")
def home():
    return "Flask ASR server running"

@app.route("/upload", methods=["POST"])
def upload_and_transcribe():
    file_path = None
    try:
        if not request.data:
            return Response("No audio received", status=400)

        file_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
        with open(file_path, "wb") as f:
            f.write(request.data)

        result = transcriber(file_path)
        text = result["text"].strip()

        return Response(text, mimetype="text/plain")

    except Exception as e:
        return Response(str(e), status=500)

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)



