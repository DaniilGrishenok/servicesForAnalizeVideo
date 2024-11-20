import os
from pydub import AudioSegment
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from flask import Flask, request, jsonify, render_template_string
import requests
import shutil
import time

app = Flask(__name__)


def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError:
        return "Ошибка сервиса распознавания речи"


def process_video_for_audio_text(video_path, audio_path):
    start_time = time.time()
    extract_audio_from_video(video_path, audio_path)
    text = audio_to_text(audio_path)
    processing_time = time.time() - start_time
    return text, processing_time


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
        <title>Привет, я Audio-to-Text сервис</title>
        <h1>Привет, я Audio-to-Text сервис</h1>
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
    audio_path = 'extracted_audio.wav'
    video_file.save(video_path)
    text, processing_time = process_video_for_audio_text(video_path, audio_path)
    # Безопасное удаление файлов после обработки
    safe_delete(video_path)
    safe_delete(audio_path)
    return jsonify({"text": text, "processing_time": processing_time})


@app.route('/process_video_url', methods=['POST'])
def process_video_url_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400

    video_path = 'downloaded_video.mp4'
    audio_path = 'extracted_audio.wav'
    download_video(video_url, video_path)
    text, processing_time = process_video_for_audio_text(video_path, audio_path)
    # Безопасное удаление файлов после обработки
    safe_delete(video_path)
    safe_delete(audio_path)
    return jsonify({"text": text, "processing_time": processing_time})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
