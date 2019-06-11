from flask import Flask, request, jsonify
from classifiers.type_classifier import TypeClassifier
from classifiers.tag_classifier import TagClassifier

app = Flask(__name__)
cat_classifier = TypeClassifier()
tag_classifier = TagClassifier()


@app.route('/category', methods=['POST'])
def classify_clothing_type():
    """Classify given image into one of clothing categories"""
    image_file = _get_image_from_request(request)
    predictions = cat_classifier.classify(image_file)
    return jsonify(predictions)

@app.route('/tag', methods=['POST'])
def classify_pattern_type():
    """Return probabilities of clothing pattern types."""
    image_file = _get_image_from_request(request)
    predictions = tag_classifier.classify(image_file)
    return jsonify(predictions)

def _get_image_from_request(received_req):
    if (received_req.files is not None 
        and 'image' in received_req.files.keys()):
        return request.files['image'].stream._file

    # TODO: Add different image sending handling
    



