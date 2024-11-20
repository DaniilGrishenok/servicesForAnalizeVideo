import os
import cv2
import pytesseract
from PIL import Image
from tqdm import tqdm
import re
from flask import Flask, request, jsonify, render_template_string
import requests
import shutil
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import difflib
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Загрузка модели генерации текста
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def extract_frames(video_path, frame_interval=5):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    video.release()
    return frames

def is_meaningful_text(text):
    clean_text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text).strip()
    words = clean_text.split()
    return len(words) > 3 and len(clean_text) > 10  # Минимальная длина осмысленной фразы

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contrast = cv2.convertScaleAbs(binary, alpha=2, beta=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, kernel)
    return opened

def extract_text_from_frames(frames):
    texts = []
    for frame in tqdm(frames, desc="Extracting text from frames with Tesseract"):
        preprocessed_image = preprocess_image(frame)
        image = Image.fromarray(preprocessed_image)
        text = pytesseract.image_to_string(image, lang='rus+eng')
        sentences = sent_tokenize(text)
        for sentence in sentences:
            filtered_text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', sentence).strip()
            filtered_text = ' '.join(filtered_text.split())
            if is_meaningful_text(filtered_text):
                texts.append(filtered_text)
    return texts

def remove_similar_texts(texts, threshold=0.8):
    unique_texts = []
    for text in texts:
        if not any(difflib.SequenceMatcher(None, text, unique_text).ratio() > threshold for unique_text in unique_texts):
            unique_texts.append(text)
    return unique_texts

def post_process_texts(texts):
    # Дополнительная пост-обработка текстов
    meaningful_texts = [text for text in texts if is_meaningful_text(text)]
    unique_texts = remove_similar_texts(meaningful_texts)
    return unique_texts

def generate_coherent_text(texts):
    unique_texts = post_process_texts(texts)
    return unique_texts

def process_video_for_text(video_path: str):
    start_time = time.time()
    frames = extract_frames(video_path)
    texts = extract_text_from_frames(frames)
    coherent_text = generate_coherent_text(texts)
    processing_time = time.time() - start_time
    return coherent_text, processing_time

def download_video(url, filename):
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def safe_delete(file_path):
    """ Удаляет файл, если он существует """
    if os.path.exists(file_path):
        os.remove(file_path)

@app.route('/')
def home():
    return render_template_string("""
        <!doctype html>
        <title>Привет, я Text-On-Video-To-Text сервис</title>
        <h1>Привет, я Text-On-Video-To-Text сервис</h1>
        <form action="/process_video" method="post" enctype="multipart/form-data">
          <input type="file" name="video">
          <input type="submit" value="Upload">
        </form>
        <br>
        <form action="/process_video_url" method="post">
          <input type="text" name="video_url" placeholder="Введите URL видео">
          <input type="submit" value="Download and Process">
        </form>
    """)

@app.route('/process_video', methods=['POST'])
def process_video_route():
    video_file = request.files['video']
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)
    coherent_text, processing_time = process_video_for_text(video_path)
    # Безопасное удаление файлов после обработки
    safe_delete(video_path)
    return jsonify({"coherent_text": coherent_text, "processing_time": processing_time})

@app.route('/process_video_url', methods=['POST'])
def process_video_url_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400

    video_path = 'downloaded_video.mp4'
    download_video(video_url, video_path)
    coherent_text, processing_time = process_video_for_text(video_path)
    # Безопасное удаление файлов после обработки
    safe_delete(video_path)
    return jsonify({"coherent_text": coherent_text, "processing_time": processing_time})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
