import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# ===========================================
#       CONFIGURACIÓN DEL DATASET
# ===========================================
data_path = "dataset"
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    subset="validation"
)

num_classes = len(train_gen.class_indices)
print("Clases detectadas:", train_gen.class_indices)

# ===========================================
#       MODELO CNN CON TRANSFER LEARNING
# ===========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*img_size, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Congelamos capas convolucionales

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===========================================
#       ENTRENAMIENTO
# ===========================================
epochs = 20

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ===========================================
#       GUARDAR MODELO
# ===========================================
model.save("face_cnn_model.h5")
print("Modelo guardado como face_cnn_model.h5")

# ===========================================
#       GRAFICAR RESULTADOS
# ===========================================
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy del Modelo")
plt.show()