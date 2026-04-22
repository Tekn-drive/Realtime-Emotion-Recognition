import cv2
import model_methods 
from model_methods import initialize_model_and_others, camera_prediction_CLI

model, emotions, facecasc = initialize_model_and_others()

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# start the webcam feed
#cap = cv2.VideoCapture('range.gif')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break
        
    results = camera_prediction_CLI(model, frame, facecasc, emotions)

    for result in results:
        x, y, w, h = result["box"]
        emotion = result["emotion"]
        confidence = result["confidence"]
        cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(500,500),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()