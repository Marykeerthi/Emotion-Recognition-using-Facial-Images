from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model("final_emotion_model.h5")
CLASS_NAMES = ["Not Happy", "Happy"]

def predict_emotion(image):
    image = image.resize((48, 48)).convert("RGB")
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array)[0][0]
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob
    return f"{label} ({confidence:.2%})"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file.stream)
            prediction = predict_emotion(img)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)