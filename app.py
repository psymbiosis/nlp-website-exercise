import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify
import numpy as np

with open('article_classifier.pkl', 'rb') as f:
    pipeline = pickle.load(f)
app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    predictions = pipeline.predict_proba([data['user_input']])
    classes = pipeline.steps[1][1].classes_
    prediction = np.argmax(predictions)
    return jsonify({'predicted class': classes[prediction], 'likelihood' : predictions[prediction]})
