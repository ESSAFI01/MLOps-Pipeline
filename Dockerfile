# Utiliser une image Python officielle légère
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires pour XGBoost
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY backend/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY backend/src/ ./src/

# Copier les données et modèles
COPY Mlpro/ ./Mlpro/

# Créer les dossiers nécessaires
RUN mkdir -p ./mlruns ./Mlpro/models ./Mlpro/dataSet

# Exposer le port pour l'API (optionnel)
EXPOSE 5000

# Variable d'environnement
ENV PYTHONUNBUFFERED=1

# Commande par défaut (peut être overridée)
CMD ["python", "src/train.py", "baseline"]
