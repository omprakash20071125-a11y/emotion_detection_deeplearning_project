import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load trained model
model = tf.keras.models.load_model(
    "emotion_detection_model.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)
# Emotion labels
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

IMG_SIZE = 224

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # Normalize
        face = face / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Prediction
        predictions = model.predict(face, verbose=0)

        emotion_index = np.argmax(predictions)

        emotion = emotion_labels[emotion_index]

        confidence = np.max(predictions)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Put text
        text = f"{emotion} ({confidence:.2f})"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()