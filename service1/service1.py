import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import open_clip as clip
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify, render_template_string
import requests
import shutil
import time
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


class Data:
    def __init__(self, id, description, tags):
        self.id = id
        self.description = description
        self.tags = tags


# Загрузка модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

# Загрузка модели BLIP для генерации описаний
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Загрузка модели перевода на русский язык
translation_model_name = "Helsinki-NLP/opus-mt-en-ru"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name).to(device)


# Загрузка видео и извлечение кадров
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


# Преобразование кадров в эмбеддинги
def frames_to_embeddings(frames, model, preprocess, device):
    embeddings = []
    for frame in tqdm(frames, desc="Processing frames"):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


# Индексация эмбеддингов в FAISS
def index_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Поиск ближайших эмбеддингов в FAISS
def search_embeddings(query_embedding, index, top_k=5):
    D, I = index.search(query_embedding, top_k)
    return I[0]


# Обработка запросов пользователя
def process_query(query, model, preprocess, index, frames, device):
    query_embedding = model.encode_text(clip.tokenize([query]).to(device)).detach().cpu().numpy()
    nearest_indices = search_embeddings(query_embedding, index)
    return [frames[i] for i in nearest_indices]


# Генерация описаний кадров
def generate_descriptions(frames, processor, model, device):
    descriptions = []
    for frame in tqdm(frames, desc="Generating descriptions"):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        descriptions.append(description)
    return descriptions


# Перевод описаний на русский язык
def translate_descriptions(descriptions, model, tokenizer, device):
    translated_descriptions = []
    for description in tqdm(descriptions, desc="Translating descriptions"):
        inputs = tokenizer(description, return_tensors="pt", padding=True).to(device)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_descriptions.append(translated_text)
    return translated_descriptions


# Функция для извлечения тегов из описания
def extract_tags(description):
    return [word.strip().lower() for word in description.split() if len(word) > 2]


def process_video(video_path: str):
    try:
        start_time = time.time()
        frames = extract_frames(video_path)
        embeddings = frames_to_embeddings(frames, model, preprocess, device)
        index = index_embeddings(embeddings)

        query = "Что происходит?"
        matching_frames = process_query(query, model, preprocess, index, frames, device)
        descriptions = generate_descriptions(matching_frames, blip_processor, blip_model, device)
        translated_descriptions = translate_descriptions(descriptions, translator_model, translator_tokenizer, device)

        results = []
        for i, (description, translated_description) in enumerate(zip(descriptions, translated_descriptions)):
            tags = extract_tags(translated_description)
            data = Data(id=i, description=translated_description, tags=tags)
            results.append(data)

        processing_time = time.time() - start_time
        return results, processing_time
    except Exception as e:
        logging.exception("An error occurred while processing the video")
        return [], 0


def download_video(url, filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise


def safe_delete(file_path):
    """ Удаляет файл, если он существует """
    if os.path.exists(file_path):
        os.remove(file_path)


@app.route('/')
def home():
    return render_template_string("""
        <!doctype html>
        <title>Привет, я CLIP-BLIP сервис</title>
        <h1>Привет, я CLIP-BLIP сервис</h1>
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
    try:
        video_file = request.files['video']
        video_path = 'uploaded_video.mp4'
        video_file.save(video_path)
        results, processing_time = process_video(video_path)
        safe_delete(video_path)
        return jsonify({"results": [result.__dict__ for result in results], "processing_time": processing_time})
    except Exception as e:
        logging.exception("An error occurred while processing the video route")
        return jsonify({"error": str(e)}), 500


@app.route('/process_video_url', methods=['POST'])
def process_video_url_route():
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        if not video_url:
            return jsonify({"error": "No video_url provided"}), 400

        video_path = 'downloaded_video.mp4'
        download_video(video_url, video_path)
        results, processing_time = process_video(video_path)
        safe_delete(video_path)
        return jsonify({"results": [result.__dict__ for result in results], "processing_time": processing_time})
    except Exception as e:
        logging.exception("An error occurred while processing the video URL route")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
