# Используем официальный образ Python
FROM python:3.8-slim

# Устанавливаем необходимые системные пакеты, включая ffmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем нужные Python-библиотеки
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Копируем файл resolv.conf для настройки DNS
COPY resolv.conf /etc/resolv.conf
# Копируем код сервиса
COPY . /app
WORKDIR /app
# Настройка DNS-серверов

# Запуск сервиса
CMD ["python", "service3.py"]
