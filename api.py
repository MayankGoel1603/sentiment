from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
import matplotlib
matplotlib.use('Agg')  # Required for cloud and server backends
import matplotlib.pyplot as plt

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords exist (avoids LookupError)
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

# Load models once at startup
with open("Models/model_xgb.pkl", "rb") as f:
    predictor = pickle.load(f)
with open("Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("Models/countVectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

def preprocess_text(text):
    """Clean, tokenize, remove stopwords, and stem."""
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = [
        stemmer.stem(word)
        for word in text.lower().split()
        if word not in STOPWORDS
    ]
    return " ".join(words)

def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = cv.transform([processed]).toarray()
    scaled = scaler.transform(vectorized)
    prediction = predictor.predict(scaled)[0]
    return "Positive" if prediction == 1 else "Negative"

@app.route("/")
def home():
    return "<h2>Sentiment API is running successfully 🚀</h2>"

@app.route("/predict", methods=["POST"])
def predict():
    try:

        # CASE 1 → File upload (CSV)
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            if "Sentence" not in data.columns:
                return jsonify({"error": "CSV must contain a 'Sentence' column"}), 400

            data["Predicted"] = data["Sentence"].apply(predict_sentiment)

            # Create pie chart
            plt.figure(figsize=(5, 5))
            data["Predicted"].value_counts().plot(kind="pie", autopct="%1.1f%%")
            img = BytesIO()
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)

            # Prepare CSV output
            output = BytesIO()
            data.to_csv(output, index=False)
            output.seek(0)

            response = send_file(
                output,
                mimetype="text/csv",
                as_attachment=True,
                download_name="predictions.csv"
            )

            # Base64 encode the chart
            response.headers["X-Graph"] = base64.b64encode(img.getvalue()).decode("utf-8")
            return response

        # CASE 2 → JSON request with text
        data = request.get_json()
        if data and "text" in data:
            result = predict_sentiment(data["text"])
            return jsonify({"result": result}), 200

        return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
