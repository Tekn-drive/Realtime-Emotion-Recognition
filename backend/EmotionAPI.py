from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from pydantic import BaseModel 
from utils.model_methods import initialize_model_and_others, process_frame 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, emotions, facecasc = initialize_model_and_others()

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

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
        
        gray, faces = process_frame(frame, facecasc)

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
                "emotion": emotions[maxindex],
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