from keras.models import load_model
import cv2
import numpy as np

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
model = load_model('./models/model.h5')
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('./data/example_video.mp4')

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = frame[y:y + h, x:x + w]

        face = np.expand_dims(cv2.resize(roi_color, (48, 48)), 0)
        pred = model.predict(face)

        prob = np.max(pred)
        label = np.argmax(pred)
        text = emotions[label] + ': ' + str(prob)

        cv2.putText(frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
