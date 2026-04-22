import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
import os 

def initialize_model_and_others():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, "model", "haarcascade_frontalface_default.xml")
    model_path = os.path.join(BASE_DIR, "model", "emotionrecognition.h5")
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    facecasc = cv2.CascadeClassifier(cascade_path)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights(model_path)
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
    