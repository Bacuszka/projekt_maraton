# Bazowy obraz Pythona 3.11 (lekki)
FROM python:3.11-slim

# Ustaw zmienną środowiskową na potrzeby Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalacja podstawowych narzędzi systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Ustaw katalog roboczy
WORKDIR /app

# Kopiuj pliki requirements.txt i instaluj zależności
COPY requirements.txt .

# Instalacja zależności Pythona
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj całą aplikację do kontenera
COPY . .

# Domyślna komenda uruchomienia aplikacji
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
