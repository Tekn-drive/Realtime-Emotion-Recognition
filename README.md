# Realtime-Emotion-Recognition
A real-time webcam-based emotion recognition system powered by deep learning, now upgraded into a full-stack web application using React and FastAPI.

Originally developed as a university computer science project (2023), this is a refined version featuring a modern web interface and improved usability.

## 🚀 New Update (v2.0)
- Migrated from CLI-based application to a React web app
- Real-time webcam emotion detection
- Bounding box face detection
- Emotion prediction with confidence score
- REST API backend for model inference

### Usage Guide
#### API:
- Make sure that you changed your directory from root to backend by doing `cd backend`
- Install all requirements from requirements.txt for your python environment (preferably Python 3.10.0)
- Launch the API by typing `uvicorn EmotionAPI:app --reload --port 8000`

#### React App:
- Make sure that you are in the root folder of this project and that the API has been launched.
- Type `npm start`
- Enjoy the app

## For those who still want to use the classical CLI app (v1.0)
No worries, the CLI app is still stored in this project. Here is how to access it:
- Make sure that the requirements in requirements.txt are installed.
- Change the directory from root to backend/utils by typing `cd backend/utils/`
- Just execute the program by typing `python CLI.py`