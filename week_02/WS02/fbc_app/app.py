import os, json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path

# Load .env variables
load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Init Azure client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-07-01-preview"
)

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'FBC_DB'
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.point(lambda x: 0 if x < 140 else 255)  # binarize
    return img

def extract_fbc(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    print("OCR Output:\n", text)  # Keep this for debugging

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fbc_data = {}
    for i, line in enumerate(lines):
        if any(char.isdigit() for char in line):  # crude way to detect value
            key = lines[i - 1] if i > 0 else f"field_{i}"
            fbc_data[key] = line
    return fbc_data

def summarize_fbc(fbc_data):
    if not fbc_data or len(fbc_data.keys()) == 1:
        return "Sorry, the image couldn't be interpreted. Please upload a clearer FBC test."
    message_text = "Patient FBC results:\n"
    for k, v in fbc_data.items():
        if k != "patient_id":
            message_text += f"- {k}: {v}\n"
    print("Message sent to OpenAI:\n", message_text)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a medical assistant who explains FBC results in simple, patient-friendly language."},
            {"role": "user", "content": message_text}
        ]
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    explanation = ""
    error = ""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file selected"
        elif not allowed_file(file.filename):
            error = "Sorry, we only accept jpg or png format"
        else:
            filename = secure_filename(file.filename)
            patient_id = os.path.splitext(filename)[0]
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            fbc_data = extract_fbc(image_path)
            fbc_data["patient_id"] = patient_id

            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{patient_id}.json"), "w", encoding="utf-8") as f:
                json.dump(fbc_data, f, indent=2)

            explanation = summarize_fbc(fbc_data)

    return render_template("index.html", explanation=explanation, error=error)

if __name__ == "__main__":
    app.run(debug=True)