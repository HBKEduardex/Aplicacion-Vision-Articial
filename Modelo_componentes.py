"""
Clasificación de componentes electrónicos con CNN + Keras Tuner.
Versión con entrenamiento sobre imágenes originales y con bordes (Canny).
Convierte los modelos a formato TFLite para despliegue en dispositivos embebidos.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras_tuner as kt

# Parámetros globales
IMG_SIZE = 128
LIMIT_ACCURACY = 0.95
DATASET_DIR = "/home/belen/Vision/data_componentes"

def load_dataset(path, use_edges=False):
    """
    Carga imágenes desde el dataset y opcionalmente aplica detección de bordes (Canny).
    Retorna: imágenes, etiquetas y nombres de clases.
    """
    images, labels = [], []
    class_names = sorted(os.listdir(path))
    for idx, class_name in enumerate(class_names):
        folder = os.path.join(path, class_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            if use_edges:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels), class_names

def prepare_data(use_edges=False):
    """
    Prepara datos de entrenamiento y validación. Normaliza los pixeles y hace split.
    """
    X, y, class_names = load_dataset(DATASET_DIR, use_edges)
    X = X / 255.0  # Normalización
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return x_train, x_val, y_train, y_val, class_names

def build_model(hp):
    """
    Construye una CNN con hiperparámetros seleccionados por Keras Tuner.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    for i in range(hp.Int("conv_blocks", 1, 3, default=2)):
        model.add(layers.Conv2D(
            filters=hp.Int(f"filters_{i}", 32, 128, step=32),
            kernel_size=hp.Choice(f"kernel_size_{i}", [3, 5]),
            activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int("dense_units", 64, 256, step=64), activation="relu"))
    model.add(layers.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(4, activation="softmax"))  # 4 clases

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

class AccuracyLimitCallback(tf.keras.callbacks.Callback):
    """
    Detiene el entrenamiento si la val_accuracy supera el umbral del 95%
    """
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("val_accuracy")
        if acc and acc > LIMIT_ACCURACY:
            print(f"\nParando entrenamiento. val_accuracy alcanzó {acc:.2f}")
            self.model.stop_training = True

def train_with_keras_tuner(x_train, y_train, x_val, y_val, version="original"):
    """
    Realiza búsqueda de hiperparámetros con Keras Tuner y entrena el mejor modelo.
    """
    tuner = kt.Hyperband(build_model,
                         objective="val_accuracy",
                         max_epochs=20,
                         factor=3,
                         directory=f"kt_{version}",
                         project_name="comp_classifier")

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    acc_limit = AccuracyLimitCallback()
    checkpoint = keras.callbacks.ModelCheckpoint(f"best_model_{version}.h5", save_best_only=True)

    # Búsqueda de hiperparámetros
    tuner.search(datagen.flow(x_train, y_train, batch_size=32),
                 epochs=20,
                 validation_data=(x_val, y_val),
                 callbacks=[early_stop, acc_limit, checkpoint])

    # Entrenamiento final con el mejor modelo
    best_model = tuner.get_best_models(1)[0]
    best_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                   validation_data=(x_val, y_val),
                   epochs=15,
                   callbacks=[early_stop, acc_limit, checkpoint])

    best_model.save(f"final_model_{version}.h5")
    return best_model

def convert_to_tflite(model_path, output_name):
    """
    Convierte un modelo Keras a formato TensorFlow Lite.
    """
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_name, 'wb') as f:
        f.write(tflite_model)
    print(f"Modelo convertido y guardado como: {output_name}")

# === Entrenamiento con imágenes originales ===
x_train_o, x_val_o, y_train_o, y_val_o, class_names = prepare_data(use_edges=False)
model_original = train_with_keras_tuner(x_train_o, y_train_o, x_val_o, y_val_o, version="4_original")
convert_to_tflite("best_model_4_original.h5", "model_original4.tflite")

# === Entrenamiento con bordes (Canny) , al final no se hiso uso de este modelo el accuracy no era el esperado en el test===
x_train_e, x_val_e, y_train_e, y_val_e, _ = prepare_data(use_edges=True)
model_edges = train_with_keras_tuner(x_train_e, y_train_e, x_val_e, y_val_e, version="4_edges")
convert_to_tflite("best_model_4_edges.h5", "model_edges4.tflite")
