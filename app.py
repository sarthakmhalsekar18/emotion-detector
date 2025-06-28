from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.config import set_visible_devices

app = Flask(__name__)

# Disable GPU
set_visible_devices([], 'GPU')

# Load your trained model
model = load_model('emotion.h5')
model.make_predict_function()  # Required for thread safety

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face detector
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
        if 'image' not in request.files:
            return render_template('index.html', emotion="No file uploaded")
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', emotion="No file selected")
            
        try:
            os.makedirs('static', exist_ok=True)
            filepath = os.path.join('static', 'uploaded_image.jpg')
            file.save(filepath)
            
            face_input = preprocess_image(filepath)
            if face_input is not None:
                prediction = model.predict(face_input)
                emotion = emotion_labels[np.argmax(prediction)]
            else:
                emotion = "No face detected"
        except Exception as e:
            emotion = f"Error: {str(e)}"

    return render_template('index.html', emotion=emotion)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production