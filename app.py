from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import requests
from io import BytesIO
import torchvision.transforms as transforms
from torchvision import models
import base64

# ======== Flask Setup ========
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ======== Text Moderation Setup ========
with open("text_moderation_model.pkl", "rb") as f:
    model_text = pickle.load(f)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def moderate_sentence(sentence):
    cleaned_text = clean_text(sentence)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model_text.predict(text_vectorized)[0]
    feature_names = np.array(vectorizer.get_feature_names_out())
    word_importance = model_text.coef_[0]
    offensive_words = []
    for word in cleaned_text.split():
        if word in feature_names:
            word_index = np.where(feature_names == word)[0][0]
            word_score = word_importance[word_index]
            if word_score > 0.6:
                offensive_words.append(word)
    words = sentence.split()
    censored_sentence = " ".join(["*" * len(word) if word.lower() in offensive_words else word for word in words])
    return censored_sentence, offensive_words

# ======== Image Moderation Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model with adjusted final layer
model_image = models.resnet18(pretrained=False)
model_image.fc = nn.Linear(model_image.fc.in_features, 2)
model_image.load_state_dict(torch.load("violence_model_resnet18.pt", map_location=device))
model_image = model_image.to(device)
model_image.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_and_blur_if_needed(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "error", "Failed to fetch image"

        image = Image.open(BytesIO(response.content)).convert('RGB')
        transformed = transform(image).unsqueeze(0).to(device)
        output = model_image(transformed)
        _, predicted = torch.max(output, 1)
        class_names = ['not_violence', 'violence']
        label = class_names[predicted.item()]

        if label == 'violence':
            image = image.filter(ImageFilter.GaussianBlur(15))

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return label, img_str

    except Exception as e:
        return "error", str(e)

# ======== Routes ========

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/moderate", methods=["POST"])
def moderate():
    data = request.json
    text = data.get("text", "")
    moderated_text, offensive_words = moderate_sentence(text)
    return jsonify({"moderated_text": moderated_text, "offensive_words": offensive_words})

@app.route("/moderate-image", methods=["POST"])
def moderate_image_route():
    data = request.json
    url = data.get("image_url", "")
    label, img_base64 = predict_and_blur_if_needed(url)
    if label == "error":
        return jsonify({"error": img_base64}), 500
    return jsonify({
        "classification": label,
        "image_data": img_base64
    })

# ======== Run the App ========
if __name__ == "__main__":
    app.run(debug=True)
