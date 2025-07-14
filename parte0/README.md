Parte 0: Entrenamiento y Serialización del Modelo
Este directorio contiene el código para entrenar y serializar el modelo de Keras, el cual implementa una Red Neuronal (CNN) para clasificar imagenes del dataset CIFAR-10.

Configuración y Ejecución
0. Instalar Poetry en la consola (en caso que no se tenga instalado previamente):
pip install poetry

1. Navega al directorio parte0:
cd parte0

2. Instala las dependencias usando Poetry:
poetry install

2.1 Activar el ambiente poetry local para volver a entrenar el modelo (sólo si se requiere)
https://python-poetry.org/docs/managing-environments/#bash-csh-zsh

bash-csh-zsh
eval $(poetry env activate)

PowerShell
Invoke-Expression (poetry env activate)

Entrenamiento (como recordatorio)
python train_model.py

3. Instala manualmente TensorFlow 2.15.0 dentro del entorno Poetry (para evitar problemas con dependencias de wheels):
poetry run pip install tensorflow==2.15.0

3.1 Instalar manualmente scikit-learn: 
poetry add scikit-learn

4. Ejecuta el script de entrenamiento:
Este comando activará el entorno virtual de Poetry y ejecutará tu script, el cual entrenará el modelo y lo guardará en la carpeta app/ (que se creará automáticamente si no existe) junto con el preprocesador.

poetry run python train_model.py