from flask import Flask, request, jsonify
from PIL import Image
app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})


if __name__ == "__main__":
    app.run(debug=True)