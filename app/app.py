
"""
Aplicación Streamlit para predicción de imágenes CIFAR-10
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Configuración de la página
st.set_page_config(
    page_title="Clasificador CIFAR-10",
    page_icon="🖼️",
    layout="wide"
)

# Nombres de las clases
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_NAMES_ES = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
                  'perro', 'rana', 'caballo', 'barco', 'camión']


@st.cache_resource
def load_model():
    """Cargamos el modelo entrenado"""
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


def preprocess_image(image_array):
    """Preprocesa la imagen para el modelo"""
    # Redimensionar a 32x32
    image_resized = tf.image.resize(image_array, [32, 32])
    # Normalizar
    image_normalized = tf.cast(image_resized, tf.float32) / 255.0
    # Agregar dimensión de batch
    image_batch = tf.expand_dims(image_normalized, 0)
    return image_batch


def create_synthetic_image(params):
    """Crea una imagen sintética basada en los parámetros"""
    # Crear imagen base
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    
    # Color dominante
    color_map = {
        'Rojo': [255, 0, 0],
        'Verde': [0, 255, 0],
        'Azul': [0, 0, 255],
        'Amarillo': [255, 255, 0],
        'Morado': [128, 0, 128],
        'Naranja': [255, 165, 0],
        'Cyan': [0, 255, 255],
        'Rosa': [255, 192, 203]
    }
    
    base_color = np.array(color_map[params['color']])
    
    # Aplicar color base con intensidad
    intensity = params['brightness'] / 100.0
    image[:, :] = (base_color * intensity).astype(np.uint8)
    
    # Agregar formas según el tipo
    if params['shape'] == 'Circular':
        # Crear forma circular
        center = (16, 16)
        radius = int(params['size'] / 10)
        y, x = np.ogrid[:32, :32]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = [255, 255, 255]  # Blanco
    
    elif params['shape'] == 'Rectangular':
        # Crear rectángulo
        size = int(params['size'] / 5)
        start = 16 - size // 2
        end = 16 + size // 2
        image[start:end, start:end] = [255, 255, 255]
    
    # Agregar textura
    if params['texture'] > 50:
        noise = np.random.randint(0, 50, (32, 32, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image


def main():
    """Función principal de la aplicación"""
    
    # Título y descripción
    st.title("🖼️ Clasificador de Imágenes CIFAR-10")
    st.markdown("""
    ### Descripción
    Esta aplicación utiliza un modelo de **Red Neuronal Convolucional (CNN)** entrenado para clasificar imágenes 
    en 10 categorías diferentes del dataset CIFAR-10.
    
    **Categorías disponibles:** avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
    
    Puedes subir una imagen o generar una imagen sintética usando los controles a continuación.
    """)
    
    # Cargar modelo
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar para controles
    st.sidebar.header("🎛️ Controles de Imagen")
    
    # Opción de entrada
    input_type = st.sidebar.radio(
        "Selecciona el tipo de entrada:",
        ["Subir imagen", "Generar imagen sintética"]
    )
    
    image_to_predict = None
    
    if input_type == "Subir imagen":
        uploaded_file = st.sidebar.file_uploader(
            "Sube una imagen",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen para clasificar"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_to_predict = image_array
    
    else:  # Generar imagen sintética
        st.sidebar.subheader("Parámetros de la imagen sintética")
        
        # 5 entradas como mínimo
        color = st.sidebar.selectbox(
            "Color dominante:",
            ['Rojo', 'Verde', 'Azul', 'Amarillo', 'Morado', 'Naranja', 'Cyan', 'Rosa']
        )
        
        brightness = st.sidebar.slider(
            "Brillo (%):",
            min_value=10,
            max_value=100,
            value=70,
            help="Controla la intensidad del color"
        )
        
        shape = st.sidebar.selectbox(
            "Forma principal:",
            ['Circular', 'Rectangular', 'Sin forma']
        )
        
        size = st.sidebar.slider(
            "Tamaño de la forma:",
            min_value=10,
            max_value=100,
            value=50,
            help="Tamaño de la forma principal"
        )
        
        texture = st.sidebar.slider(
            "Textura/Ruido (%):",
            min_value=0,
            max_value=100,
            value=20,
            help="Cantidad de textura añadida"
        )
        
        # Generar imagen sintética
        params = {
            'color': color,
            'brightness': brightness,
            'shape': shape,
            'size': size,
            'texture': texture
        }
        
        image_to_predict = create_synthetic_image(params)
    
    # Mostrar imagen y predicción
    if image_to_predict is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagen a clasificar")
            st.image(image_to_predict, caption="Imagen de entrada", width=300)
        
        with col2:
            st.subheader("🎯 Predicción del modelo")
            
            # Preprocesar y predecir
            processed_image = preprocess_image(image_to_predict)
            predictions = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Mostrar resultado
            st.success(f"**Predicción:** {CLASS_NAMES_ES[predicted_class].title()} ({CLASS_NAMES[predicted_class]})")
            st.info(f"**Confianza:** {confidence:.2%}")
            
            # Mostrar todas las probabilidades
            st.subheader("📊 Probabilidades por clase")
            
            prob_data = []
            for i, (name_es, name_en) in enumerate(zip(CLASS_NAMES_ES, CLASS_NAMES)):
                prob_data.append({
                    'Clase': f"{name_es.title()} ({name_en})",
                    'Probabilidad': predictions[0][i]
                })
            
            # Ordenar por probabilidad
            prob_data.sort(key=lambda x: x['Probabilidad'], reverse=True)
            
            # Mostrar top 5
            for i, item in enumerate(prob_data[:5]):
                st.write(f"{i+1}. **{item['Clase']}**: {item['Probabilidad']:.2%}")
    
    else:
        st.info("👆 Sube una imagen o configura los parámetros para generar una imagen sintética")
    
    # Información adicional
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Información del modelo
    - **Arquitectura:** Red Neuronal Convolucional (CNN)
    - **Dataset de entrenamiento:** CIFAR-10
    - **Resolución de entrada:** 32x32 píxeles
    - **Número de clases:** 10
    - **Framework:** TensorFlow/Keras
    """)


if __name__ == "__main__":
    main()