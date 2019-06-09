from flask import Flask, request
from classifiers.type_classifier import TypeClassifier

app = Flask(__name__)
cat_classifier = TypeClassifier()


@app.route('/category', methods=['GET', 'POST'])
def classify_clothing_type():
    """Classify given image into one of clothing categories"""
    image_file = request.files['image']

    predictions = cat_classifier.classify(image_file)
    print(predictions)
    return "Testy"

