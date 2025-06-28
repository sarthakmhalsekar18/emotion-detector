from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model('emotion.h5')

# Emotion labels - change if your model uses different ones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Use OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join('Static', 'uploaded_image.jpg')
        file.save(filepath)

        face_input = preprocess_image(filepath)
        if face_input is not None:
            prediction = model.predict(face_input)
            emotion = emotion_labels[np.argmax(prediction)]
        else:
            emotion = "No face detected"

    return render_template('index.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
