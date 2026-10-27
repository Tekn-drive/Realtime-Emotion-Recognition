import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os 

def initialize_model_and_others():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, "model", "haarcascade_frontalface_default.xml")
    model_path = os.path.join(BASE_DIR, "model", "emotion_classifier.keras")
    emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
    facecasc = cv2.CascadeClassifier(cascade_path)
    model = load_model(model_path)
    return model, emotion_dict, facecasc

def process_frame(frame, facecasc):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    return gray, faces

def camera_prediction_CLI(model, frame, facecasc, emotion_dict):
    results = []
    gray, faces = process_frame(frame, facecasc)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cropped_img.astype("float32") / 255.0
        cropped_img = np.expand_dims(cropped_img, axis=(0, -1))
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        results.append({"box": (x, y, w, h), "emotion": emotion_dict[maxindex], "confidence": confidence})
    
    return results
    