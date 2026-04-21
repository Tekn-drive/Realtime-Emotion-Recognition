from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
import base64
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
model.load_weights('Utils/Model/emotionrecognition.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary mapping class labels with corresponding emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

facecasc = cv2.CascadeClassifier('Utils/Model/haarcascade_frontalface_default.xml')
print("Face cascade loaded successfully")

class frameInput(BaseModel):
    frame: str

@app.post("/predict_emotion")
def predict_emotion(data: frameInput):
    try:
        base64_data = data.frame.split(",")[1]

        # Decode base64 → bytes
        image_bytes = base64.b64decode(base64_data)

        # Convert bytes → numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid frame received"}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        if faces is None or len(faces) == 0:
            return {"error": "No faces detected"}
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype("float32") / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=(0, -1))
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            return {
                "emotion": emotion_dict[maxindex],
                "confidence": confidence,
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            }
    except Exception as e:
        return {"error": str(e)}