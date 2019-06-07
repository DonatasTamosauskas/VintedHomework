from flask import Flask
app = Flask(__name__)


@app.route('/category')
def classify_clothing_type():
    """Classify given image into one of clothing categories"""
    return "Testy"

