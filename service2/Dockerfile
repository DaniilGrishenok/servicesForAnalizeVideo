# Используем официальный образ Python
FROM python:3.8

# Устанавливаем необходимые системные пакеты, включая Tesseract и библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем нужные Python-библиотеки
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Настройка DNS-серверов
# Копируем файл resolv.conf для настройки DNS
COPY resolv.conf /etc/resolv.conf
# Копируем код сервиса
COPY . /app
WORKDIR /app

# Запуск сервиса
CMD ["python", "service2.py"]
