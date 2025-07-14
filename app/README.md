"# Clasificador CIFAR-10 - Streamlit App" 
Este directorio contiene la aplicación streamlit y los archivos para su dockerización.

# Configuración y Ejecución de la Aplicación Streamlit

0. Instalar Poetry: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
1. Navega al directorio app: cd app
2. Instala las dependencias usando Poetry: poetry install
3. Activar el entorno virtual: poetry env activate
4. Ejecuta la aplicación Streamlit: streamlit run app.py
5. Asegúrate de estar en el directorio app: cd app

Abre tu navegador en: http://localhost:8501

Nota: En caso que se requiere volver a entrenar el modelo: 
1. poetry run python train_model.py
2. Instalación de las dependencias: librerias
 * python = ">=3.11,<3.13"
 * tensorflow = "2.16.1"
 * streamlit = "^1.28.0"
 * pillow = "^10.0.0"
 * joblib = "^1.3.0"
 * numpy = "^1.26.0"
 * resumen: poetry add tensorflow-cpu==2.16.1 streamlit pillow joblib numpy

# Construcción y Ejecución de la Imagen Docker

0. Navega al directorio app: cd app
1. Construir la imagen Docker: docker build -t cifar10-classifier .
2. Ejecuta el contenedor Docker, mapeando el puerto 8501: docker run -p 8501:8501 -d --name mi-clasificador cifar10-classifier
2.1 verificamos el estado: docker ps
3. Abre tu navegador en: http://localhost:8501


Nota: Dependencias compatibles
* tensorflow==2.13.0
* streamlit==1.28.1
* numpy==1.24.3
* matplotlib==3.7.2
* seaborn==0.12.2
* scikit-learn==1.3.0
* Pillow==10.0.1
* plotly==5.17.0
* joblib==1.3.2
* opencv-python-headless==4.8.1.78