from flask import Flask, render_template, request
import os
import pandas as pd
from transformers import pipeline
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    messages = df['message'].astype(str).tolist()
    results = sentiment_model(messages)
    labels = [r['label'] for r in results]

    # Count and determine project health
    # Count and determine project health
    count = Counter(labels)

    # Extract counts
    anger = count['Anger']
    frustration = count['Frustration']
    joy = count['Joy']
    satisfaction = count['Satisfaction']
    excitement = count['Excitement']
    sadness = count['Sadness']
    neutral = count['Neutral']

    positive = joy + satisfaction + excitement
    negative = anger + frustration + sadness

    # Determine health
    if negative > positive:
        health = "Unstable"
    elif neutral >= max(positive, negative):
        health = "Stable"
    else:
        health = "Healthy"


    return render_template('index.html', project_health=health, results=zip(messages, labels))


if __name__ == '__main__':
    app.run(debug=True)
