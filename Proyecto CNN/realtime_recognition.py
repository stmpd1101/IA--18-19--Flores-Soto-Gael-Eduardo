import cv2
import numpy as np
import tensorflow as tf
import os

# Cargar modelo entrenado
model = tf.keras.models.load_model("face_cnn_model.h5")

# Clases
class_names = sorted(os.listdir("dataset"))

# Detector facial de OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

print("Presiona Q para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224,224))
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        # Predicción
        predictions = model.predict(face_input)
        class_index = np.argmax(predictions)
        class_name = class_names[class_index]
        prob = predictions[0][class_index]

        # Dibujar caja y etiqueta
        label = f"{class_name}: {prob*100:.1f}%"
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Reconocimiento Facial - CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()