import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime
import pymongo

# Load the Keras model
model = load_model('model.h5')

# OpenCV video capture
video = cv2.VideoCapture(0)

# Haar cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels dictionary
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# MongoDB connection setup
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Adjust URL and port if necessary
db = client["percobaan1"]
collection = db["streamlit"]

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        
        # Perform prediction
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Prepare data for MongoDB
        emotion_data = {
            "emotion": labels_dict[label],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
        
        # Insert into MongoDB collection
        collection.insert_one(emotion_data)
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Check for 'q' key press to exit
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
