import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch, HyperParameters, Objective
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter

# Ruta donde están las imágenes organizadas en carpetas por clase
EXTRACT_PATH = '/home/belen/Vision/Pequeno Datasetvision'
IMG_SIZE = 224
MICRO_CLASSES = ['Arduino-Uno', 'ESP32', 'Rasp', 'STM32', 'Tiva']
CLASS_COUNT = len(MICRO_CLASSES)
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(MICRO_CLASSES)}
image_paths = []
labels = []
for class_name in MICRO_CLASSES:
    class_dir = os.path.join(EXTRACT_PATH, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(CLASS_NAME_TO_ID[class_name])

# División estratificada en conjuntos de entrenamiento y validación (80%-20%)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Calcular pesos de clase para balancear el entrenamiento según la frecuencia de cada clase
counts = Counter(train_labels)
total = sum(counts.values())
class_weight = {i: total / (counts.get(i, 1) * CLASS_COUNT) for i in range(CLASS_COUNT)}

# Generador personalizado para cargar imágenes, aplicar aumento de datos y producir batches
class AugmentedGenerator(Sequence):
    def __init__(self, image_paths, labels, class_weight, batch_size=32, shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        # Número total de batches por epoch
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        # Mezclar índices al finalizar cada epoch para aleatorizar batches
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def __getitem__(self, index):
        # Obtener un batch dado el índice
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_batch, label_batch, sample_weights = [], [], []

        for i in idxs:
            image = cv2.imread(self.image_paths[i])
            if image is None:
                continue
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # Aumentos de datos probabilísticos: flip horizontal, blur gaussiano y variación de brillo
            if self.augment:
                if np.random.rand() < 0.5:
                    image = cv2.flip(image, 1)
                if np.random.rand() < 0.3:
                    image = cv2.GaussianBlur(image, (3, 3), 0)
                if np.random.rand() < 0.4:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    factor = 0.5 + np.random.uniform()
                    hsv[..., 2] = np.clip(hsv[..., 2]*factor, 0, 255)
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Normalización de pixeles a rango [0, 1]
            image = image / 255.0
            image_batch.append(image)
            label_batch.append(self.labels[i])
            sample_weights.append(self.class_weight.get(self.labels[i], 1.0))

        return np.array(image_batch), np.array(label_batch), np.array(sample_weights)

# Crear generadores para entrenamiento (con aumento) y validación (sin aumento)
train_gen = AugmentedGenerator(train_imgs, train_labels, class_weight=class_weight, augment=True)
val_gen = AugmentedGenerator(val_imgs, val_labels, class_weight=class_weight, shuffle=False)

# Función para construir modelo con hiperparámetros a optimizar
def build_model(hp):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    # Opción para agregar capa densa con hiperparámetros para unidades, activación, regularización y dropout
    if hp.Boolean("add_dense"):
        x = Dense(
            hp.Int("dense_units", 64, 512, step=64),
            activation=hp.Choice("activation", ["relu", "swish", "elu"]),
            kernel_regularizer=l2(hp.Float("l2", 1e-6, 1e-3, sampling="log"))
        )(x)
        x = Dropout(hp.Float("dropout_rate", 0.3, 0.6, step=0.1))(x)

    outputs = Dense(CLASS_COUNT, activation='softmax')(x)
    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.get(hp.Choice("optimizer", ["adam", "rmsprop"]))
    optimizer.learning_rate = hp.Float("lr", 1e-5, 1e-2, sampling="log")

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Setup del tuner para búsqueda aleatoria de hiperparámetros con Early Stopping
tuner = RandomSearch(
    build_model,
    objective=Objective("val_accuracy", direction="max"),
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='microcontroller_classifier'
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Inicio de la búsqueda y entrenamiento inicial
tuner.search(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stop])
best_model = tuner.get_best_models(num_models=1)[0]

# Fine-tuning: descongelar últimas 20 capas para entrenamiento
base_model = best_model.layers[1]
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento final con Early Stopping
history = best_model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stop])

# Guardar el modelo
best_model.save("modelo_microcontroladores3.h5")
print("Modelo guardado como modelo_microcontroladores3.h5")

# Conversión a formato TensorFlow Lite 
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
with open("modelo_microcontroladores3.tflite", "wb") as f:
    f.write(tflite_model)
print("Modelo convertido y guardado como modelo_microcontroladores3.tflite")
