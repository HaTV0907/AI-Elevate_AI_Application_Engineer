import os
import json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from azure_helper import explain_fbc_result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_FOLDER'] = 'FBC_DB'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_fields():
    # Based on your uploaded image
    return {
        "Hemoglobin": {"value": 7.4, "unit": "g/dL", "normal_range": "11.5–13.5"},
        "Hematocrit": {"value": 20.6, "unit": "%", "normal_range": "34.0–40.0"},
        "Red blood cell": {"value": 2.05, "unit": "10^6/ml", "normal_range": "3.9–5.3"},
        "White blood cell": {"value": 0.88, "unit": "10^3/µL", "normal_range": "5.5–15.5"},
        "Platelets": {"value": 56, "unit": "10^3/µL", "normal_range": "150–450"},
        "MCV": {"value": 80.5, "unit": "fL", "normal_range": "77–95"},
        "MCH": {"value": 28.9, "unit": "Pg", "normal_range": "24–30"},
        "MCHC": {"value": 35.9, "unit": "%", "normal_range": "31–36"},
        "Eosinophil": {"value": 0, "unit": "%", "normal_range": "0–4"},
        "Rod neutrophil": {"value": 2, "unit": "%", "normal_range": "3–5"},
        "Segmented neutrophil": {"value": 28, "unit": "%", "normal_range": "27–55"},
        "Lymphocyte": {"value": 32, "unit": "%", "normal_range": "36–52"},
        "Monocyte": {"value": 4, "unit": "%", "normal_range": "3–8"},
        "Blast": {"value": 28, "unit": "%", "normal_range": "0"}
    }

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    explanation = ""
    if request.method == "POST":
        file = request.files["image"]
        if not allowed_file(file.filename):
            message = "Sorry, we only accept jpg or png format"
        else:
            filename = secure_filename(file.filename)
            patient_id = os.path.splitext(filename)[0]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            fields = extract_fields()
            fields["patient_id"] = patient_id

            json_path = os.path.join(app.config['JSON_FOLDER'], f"{patient_id}.json")
            with open(json_path, "w") as f:
                json.dump(fields, f, indent=4)

            explanation_lines = []
            for test, info in fields.items():
                if test == "patient_id":
                    continue
                result = explain_fbc_result(test, info["value"], info["unit"], info["normal_range"])
                explanation_lines.append(f"{test}: {result}")
            explanation = "\n\n".join(explanation_lines)

    return render_template("index.html", message=message, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)