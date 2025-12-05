import cv2
import os

# Nombre de la persona o categoría
person_name = input("Nombre de la categoría/clase: ")

# Crear carpeta si no existe
data_path = "dataset"
person_path = os.path.join(data_path, person_name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
max_images = 200   # Puedes cambiar a mínimo 100 por persona

print("Capturando imágenes... presiona Q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        cv2.imwrite(f"{person_path}/face_{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capturando...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print("Finalizado. Imágenes guardadas en:", person_path)
