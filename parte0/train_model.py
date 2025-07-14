import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -*- coding: utf-8 -*-
"""model.keras.ipynb

Entrenamiento del modelo CNN para CIFAR-10
# Desarrollo trabajo N°2. Plataformas para Machine Learning
### Felipe Cáceres Caro
"""

# ----
# Importamos las librerías necesarias
# ----
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import joblib

# Configuramos la semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# ----
# Cargar y preparar los datos CIFAR-10
# ----
(x_train_images, y_train_labels), (x_test_images, y_test_labels) = keras.datasets.cifar10.load_data()

# Normalizamos los valores de píxeles para que estén entre 0 y 1
x_train_images = x_train_images.astype('float32') / 255.0
x_test_images = x_test_images.astype('float32') / 255.0

# Definimos los nombres de las 10 categorías del dataset CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

# ----
# Preprocesamiento de etiquetas
# ----
# Transformamos las etiquetas en codificaciones one-hot
y_train = tf.one_hot(y_train_labels.squeeze().astype(np.int32), depth=10)
y_test = tf.one_hot(y_test_labels.squeeze().astype(np.int32), depth=10)

# ----
# Dividir datos de entrenamiento para validación
# ----
# CORRECCIÓN: Convertir tensores a numpy antes de usar train_test_split
y_train_numpy = y_train.numpy()

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_images, y_train_numpy, test_size=0.2, random_state=42, stratify=y_train_labels
)

# ----
# Definir parámetros del modelo
# ----
batch_size = 32
num_classes = 10
epochs = 50

# ----
# Crear el modelo CNN
# ----
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train_images.shape[1:], activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06),
    loss='categorical_crossentropy', metrics=['accuracy'])

# ----
# Entrenamiento del modelo
# ----
history = model.fit(
    x_train_split, y_train_split,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val_split, y_val_split),
    verbose=1
)

# ----
# Evaluación del modelo
# ----
test_loss, test_accuracy = model.evaluate(x_test_images, y_test, verbose=0)

y_pred_probs = model.predict(x_test_images, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# ----
# Guardar modelo y metadatos
# ----
# CORRECCIÓN: Crear directorio app en la ubicación actual
output_dir = 'app'
os.makedirs(output_dir, exist_ok=True)

model_save_path = os.path.join(output_dir, 'model.keras')
model.save(model_save_path, save_format="keras")

metadata = {
    'class_names': class_names,
    'input_shape': (32, 32, 3),
    'num_classes': num_classes,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'epochs_trained': epochs,
    'batch_size': batch_size
}

metadata_save_path = os.path.join(output_dir, 'model_metadata.joblib')
joblib.dump(metadata, metadata_save_path)

history_save_path = os.path.join(output_dir, 'training_history.joblib')
joblib.dump(history.history, history_save_path)

print(f"Modelo guardado en: {model_save_path}")
print(f"Precisión en test: {test_accuracy:.4f}")