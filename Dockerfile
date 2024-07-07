# Utiliser une image de base officielle de Python
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip

# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet dans le répertoire de travail
COPY . .

# Télécharger et installer YOLOv5 à partir du zip
RUN curl -L https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip -o yolov5.zip \
    && unzip yolov5.zip \
    && cd yolov5-master \
    && pip install -r requirements.txt

# Exposer le port sur lequel l'application Flask s'exécute
EXPOSE 5000

# Définir la commande pour exécuter l'application Flask
CMD ["flask", "run", "--host=0.0.0.0"]
