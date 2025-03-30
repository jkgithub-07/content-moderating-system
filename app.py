from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load the trained model and vectorizer
with open("text_moderation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(sentence):
    words = sentence.split()  # Tokenize sentence
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return " ".join(filtered_words)

# Function to detect and censor offensive words
def moderate_sentence(sentence):
    cleaned_text = clean_text(sentence)  # Remove stopwords
    text_vectorized = vectorizer.transform([cleaned_text])  # Convert to vector
    prediction = model.predict(text_vectorized)[0]  # Predict if sentence is offensive

    # Get words and their importance
    feature_names = np.array(vectorizer.get_feature_names_out())
    word_importance = model.coef_[0]  # Coefficients for offensive class

    # Find offensive words
    offensive_words = []
    for word in cleaned_text.split():
        if word in feature_names:
            word_index = np.where(feature_names == word)[0][0]  # Get index in vectorizer
            word_score = word_importance[word_index]  # Get model weight
            if word_score > 0.6:  # **Threshold for offensive words**
                offensive_words.append(word)

    # Censor offensive words in the original sentence
    words = sentence.split()
    censored_sentence = " ".join(["*" * len(word) if word.lower() in offensive_words else word for word in words])

    return censored_sentence, offensive_words

@app.route("/")
def home():
    path = os.path.join(app.template_folder, "index.html")
    if not os.path.exists(path):
        return f"Template not found at: {path}", 404
    return render_template("index.html")

@app.route("/moderate", methods=["POST"])
def moderate():
    data = request.json
    text = data.get("text", "")
    moderated_text, offensive_words = moderate_sentence(text)
    return jsonify({"moderated_text": moderated_text, "offensive_words": offensive_words})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
