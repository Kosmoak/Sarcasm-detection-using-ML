from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import chardet

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open("sarcasm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("count_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["user_input"]
    data = cv.transform([user_input]).toarray()
    probabilities = model.predict_proba(data)[0]
    sarcasm_prob = probabilities[1] * 100  # Probability of sarcasm
    return render_template("index.html", prediction=f"Sarcasm ({sarcasm_prob:.2f}%)")

@app.route("/upload", methods=["POST"])
def upload_file():
    
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    
    if file.filename == "":
        return "No selected file"
    
    if file:
        try:
            lines = file.read().decode("utf-8").split("\n")
        except UnicodeDecodeError:
            file.seek(0)  # Reset the file pointer
            lines = file.read().decode("latin-1").split("\n")
        
        # Debug: Print the extracted lines
        print("Extracted lines from the file:")
        print(lines)

        predictions = []
        for line in lines:
            if line.strip():  # Skip empty lines
                # Debug: Print each line being processed
                print(f"Processing line: {line.strip()}")
                data = cv.transform([line.strip()]).toarray()
                
                probabilities = model.predict_proba(data)[0]
                sarcasm_prob = probabilities[1] * 100  # Probability of sarcasm
                predictions.append({
                    "text": line,
                    "sarcasm_prob": sarcasm_prob,
                })
        # Debug: Print predictions
        print("Predictions:")
        print(predictions)

        return render_template("results.html", predictions=predictions)
    return "File processing failed. Please try again."
if __name__ == "__main__":
    app.run(debug=True)
