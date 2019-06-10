from flask import Flask, request, jsonify
from classifiers.type_classifier import TypeClassifier

app = Flask(__name__)
cat_classifier = TypeClassifier()


@app.route('/category', methods=['GET', 'POST'])
def classify_clothing_type():
    """Classify given image into one of clothing categories"""
    image_file = _get_image_from_request(request)
    predictions = cat_classifier.classify(image_file)
    print("(Incoming image file was predicted as {}.".format(predictions))
    return jsonify(predictions)

def _get_image_from_request(received_req):
    if (received_req.files is not None 
        and 'image' in received_req.files.keys()):
        return request.files['image'].stream._file

    # TODO: Add different image sending handling
    



