FROM python:3

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

 # Copiar pyproject.toml y poetry.lock primero para aprovechar el cache 
COPY pyproject.toml poetry.lock ./
COPY model.keras ./
COPY preprocessor.joblib ./
COPY app.py ./
COPY README.md ./

# Instalar Poetry
RUN pip install poetry

# Instalar las dependencias del proyecto usando Poetry
# `--no-root` para no instalar el propio paquete del proyecto (ya que no es una librería)
# `--no-dev` para no instalar dependencias de desarrollo
RUN poetry install

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit al iniciar el contenedor
# Usamos poetry run para asegurar que se use el entorno virtual de poetry
ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
