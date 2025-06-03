import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence
from keras_tuner import Objective, RandomSearch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# Ruta principal del dataset extraído
EXTRACT_PATH = 'componentes_dataset'
IMG_SIZE = 224  # Tamaño al que se redimensionan las imágenes
MICROCONTROLADORES = ['Arduino-Uno', 'ESP32', 'Raspberry', 'STM32', 'Tiva']
ROOT_DIR = os.path.join(EXTRACT_PATH, 'train')


class YoloBinaryGenerator(Sequence):
    """
    Generador de datos personalizado basado en Keras Sequence para tareas de
    clasificación binaria y regresión de bounding boxes a partir de imágenes
    y etiquetas en formato YOLO.
    
    Parámetros:
    - image_paths: lista con rutas de imágenes
    - label_paths: lista con rutas de archivos de etiquetas (YOLO format)
    - class_weight: diccionario con pesos para cada clase, usado para balancear la pérdida
    - batch_size: tamaño de lote para el entrenamiento
    - shuffle: si se barajan las muestras después de cada época
    - augment: (opcional) aplicar aumentos de datos en las imágenes (no implementado en esta versión)
    """

    def __init__(self, image_paths, label_paths, class_weight, batch_size=4, shuffle=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        """
        Devuelve el número total de batches por época
        """
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Se llama al finalizar cada época para barajar los índices si shuffle está activo.
        """
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def __getitem__(self, index):
        """
        Genera un batch de datos (imágenes, etiquetas de clase binarias y bounding boxes)
        para el índice dado.

        Retorna:
        - batch de imágenes normalizadas
        - diccionario con etiquetas para la clasificación binaria y bounding boxes
        - diccionario con pesos para cada salida (para balancear clases)
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_batch, class_batch, bbox_batch = [], [], []

        for i in indexes:
            try:
                # Leer imagen y redimensionar
                image = cv2.imread(self.image_paths[i])
                if image is None:
                    continue
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0

                # Leer etiqueta YOLO
                label_path = self.label_paths[i]
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    parts = lines[0].strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(float(parts[0]))

                    # Clasificación binaria: microcontroladores (0-4) = 1, resto = 0
                    class_label = 1 if class_id <= 4 else 0

                    # Calcular bounding box en coordenadas absolutas
                    x_center, y_center, width, height = map(float, parts[1:])
                    xmin = (x_center - width / 2) * IMG_SIZE
                    ymin = (y_center - height / 2) * IMG_SIZE
                    xmax = (x_center + width / 2) * IMG_SIZE
                    ymax = (y_center + height / 2) * IMG_SIZE

                image_batch.append(image)
                class_batch.append(class_label)
                bbox_batch.append([xmin, ymin, xmax, ymax])
            except Exception as e:
                print(f"Error en muestra {self.image_paths[i]}: {e}")
                continue

        if len(image_batch) == 0:
            # Si no hay datos válidos, devolver arrays vacíos para no detener el entrenamiento
            return np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), {
                "class_output": np.zeros((1,)), "bbox_output": np.zeros((1, 4))
            }, {
                "class_output": np.ones((1,)), "bbox_output": np.ones((1,))
            }

        # Pesos para balancear la pérdida en función de la clase
        sample_weights = np.array([self.class_weight.get(c, 1.0) for c in class_batch])

        return np.array(image_batch), {
            "class_output": np.array(class_batch),
            "bbox_output": np.array(bbox_batch)
        }, {
            "class_output": sample_weights,
            "bbox_output": np.ones_like(sample_weights)
        }


# --- PREPARAR DATOS ---
print("Cargando rutas de datos...")

image_dir = os.path.join(ROOT_DIR, 'images')
label_dir = os.path.join(ROOT_DIR, 'labels')

# Listar todas las imágenes .jpg y sus etiquetas correspondientes
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_paths = [os.path.join(label_dir, f.replace('.jpg', '.txt')) for f in os.listdir(image_dir) if f.endswith('.jpg')]


def extract_binary_labels(label_paths):
    """
    Extrae etiquetas binarias (1 para microcontroladores, 0 para otros) de
    las anotaciones en formato YOLO.

    Parámetro:
    - label_paths: lista de rutas a archivos de etiquetas

    Retorna:
    - Lista de etiquetas binarias
    """
    binary_labels = []
    for lbl in label_paths:
        try:
            with open(lbl, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                class_id = int(float(lines[0].strip().split()[0]))
                binary_labels.append(1 if class_id <= 4 else 0)
        except:
            continue
    return binary_labels


# Separar datos en entrenamiento y validación
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42
)

# Extraer etiquetas binarias para calcular pesos de clase
train_binary_labels = extract_binary_labels(train_lbls)
counts = Counter(train_binary_labels)
total = sum(counts.values())

# Calcular pesos inversamente proporcionales a la frecuencia de cada clase
class_weight = {c: total / (v * 2) for c, v in counts.items()}
print("Pesos de clase:", class_weight)

# Crear generadores para entrenamiento y validación
train_gen = YoloBinaryGenerator(train_imgs, train_lbls, class_weight=class_weight, batch_size=8)
val_gen = YoloBinaryGenerator(val_imgs, val_lbls, class_weight=class_weight, batch_size=8, shuffle=False)


# --- MODELO BINARIO ---
def build_binary_model(hp):
    """
    Construye y compila un modelo basado en MobileNetV2 para clasificación binaria
    y predicción de bounding boxes.

    Parámetro:
    - hp: objeto HyperParameters de Keras Tuner para búsqueda de hiperparámetros

    Retorna:
    - modelo compilado de Keras listo para entrenamiento
    """
    # Base MobileNetV2 preentrenada en ImageNet sin la capa superior
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Congelar pesos base

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    # Opcional: agregar capa densa para mayor capacidad
    if hp.Boolean("add_dense"):
        x = Dense(hp.Int("dense_units", 64, 512, step=64),
                  activation=hp.Choice("activation", ["relu", "swish", "elu"]))(x)
        x = Dropout(hp.Float("dropout_rate", 0.0, 0.5, step=0.1))(x)

    # Salida binaria (sigmoid) y salida para bounding box (regresión)
    class_output = Dense(1, activation='sigmoid', name='class_output')(x)
    bbox_output = Dense(4, activation='linear', name='bbox_output')(x)

    model = Model(inputs=inputs, outputs=[class_output, bbox_output])

    # Selección de optimizador y tasa de aprendizaje
    optimizer = tf.keras.optimizers.get(hp.Choice("optimizer", ["adam", "rmsprop", "sgd"]))
    optimizer.learning_rate = hp.Float("lr", 1e-5, 1e-2, sampling="log")

    model.compile(
        optimizer=optimizer,
        loss={
            "class_output": "binary_crossentropy",
            "bbox_output": "mse"
        },
        metrics={
            "class_output": "accuracy",
            "bbox_output": "mse"
        }
    )
    return model


# --- KERAS TUNER ---
tuner = RandomSearch(
    build_binary_model,
    objective=Objective("val_class_output_accuracy", direction="max"),
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='binary_detector'
)

print("Buscando mejores hiperparámetros...")
tuner.search(train_gen, validation_data=val_gen, epochs=5)

# --- ENTRENAMIENTO FINAL CON MEJORES HIPERPARÁMETROS ---
best_model = tuner.get_best_models(1)[0]
print("Entrenando mejor modelo...")
history = best_model.fit(train_gen, validation_data=val_gen, epochs=20)

