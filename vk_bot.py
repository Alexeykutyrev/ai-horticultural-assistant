#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
from vk_api.upload import VkUpload

import logging
import httpx
import os
import json
import base64
import cv2
import numpy as np
import asyncio
import tempfile
import shutil
import time
import re
import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import pickle

# ========== ПРОВЕРКА ЗАВИСИМОСТЕЙ RAG ==========
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import PyPDF2
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG не доступен: установите sentence-transformers, faiss-cpu, pypdf2")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ================== КОНФИГУРАЦИЯ (ЗАМЕНИТЕ НА СВОИ ЗНАЧЕНИЯ) ==================
VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN", "ВАШ_ТОКЕН_ВК")          # УДАЛИТЬ реальный токен
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")                     # УДАЛИТЬ реальный ключ

LLM_API_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "qwen2.5-vl-7b-instruct"
ADMIN_ID = 541120018
ADMIN_USERNAME = "alexeykutyrev"

# ================== ДИРЕКТОРИИ ==================
USER_IMAGES_DIR = "user_data/images"
USER_ANALYSES_DIR = "user_data/analyses"
USER_DOCS_DIR = "user_data/documents"
RAG_INDEX_DIR = "user_data/rag_index"
USER_DATASETS_DIR = "user_data/datasets"
os.makedirs(USER_IMAGES_DIR, exist_ok=True)
os.makedirs(USER_ANALYSES_DIR, exist_ok=True)
os.makedirs(USER_DOCS_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(USER_DATASETS_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "conf": 0.15, "iou": 0.45, "max_det": 300, "device": "cpu",
    "video_fps": 60, "video_max_frames": 500, "return_video": True,
    "track": True, "tracker_type": "bytetrack",
    "track_high_thresh": 0.5, "track_low_thresh": 0.1,
    "new_track_thresh": 0.6, "match_thresh": 0.8
}

TASKS = {
    "🔍 Распознавание объектов": "detection",
    "✂️ Сегментация экземпляров": "instance_seg",
    "🧩 Семантическая сегментация": "semantic_seg",
    "🏷️ Классификация": "classification",
    "🔑 Распознавание точек": "keypoint"
}

PHASES = {
    "🌱 Всходы": "seedling",
    "🌿 Вегетация": "vegetation",
    "🌸 Цветение": "flowering",
    "🍎 Плодоношение": "fruiting",
    "🍂 Созревание": "ripening",
    "🌾 Уборка": "harvest"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

vk_session = vk_api.VkApi(token=VK_GROUP_TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
upload = VkUpload(vk_session)

user_data = {}

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def get_user_state(user_id):
    if user_id not in user_data:
        user_data[user_id] = {"settings": DEFAULT_SETTINGS.copy(), "step": None}
    return user_data[user_id]

def send_message(user_id, text, keyboard=None):
    params = {"user_id": user_id, "message": text, "random_id": get_random_id()}
    if keyboard:
        params["keyboard"] = keyboard
    vk.messages.send(**params)

def send_photo(user_id, photo_path, caption="", keyboard=None):
    photo = upload.photo_messages(photo_path)[0]
    attachment = f"photo{photo['owner_id']}_{photo['id']}"
    params = {"user_id": user_id, "attachment": attachment, "message": caption, "random_id": get_random_id()}
    if keyboard:
        params["keyboard"] = keyboard
    vk.messages.send(**params)

def send_document(user_id, file_path, title="file"):
    doc = upload.document_message(file_path, title=title, peer_id=user_id)
    attachment = f"doc{doc[0]['owner_id']}_{doc[0]['id']}"
    vk.messages.send(user_id=user_id, attachment=attachment, random_id=get_random_id())

def get_photo_bytes_from_attachments(attachments):
    for att in attachments:
        if att.get('type') == 'photo':
            sizes = att['photo'].get('sizes', [])
            if not sizes:
                continue
            max_size = max(sizes, key=lambda s: s.get('height', 0) * s.get('width', 0))
            url = max_size.get('url')
            if url:
                response = vk_session.http.get(url)
                return response.content
    return None

def save_user_image(user_id, img_bytes, suffix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(USER_IMAGES_DIR, f"{user_id}_{ts}_{suffix}.jpg")
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

def save_analysis(user_id, data):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(USER_ANALYSES_DIR, f"{user_id}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def get_user_analyses(user_id, limit=10):
    files = [f for f in os.listdir(USER_ANALYSES_DIR) if f.startswith(f"{user_id}_") and f.endswith(".json")]
    files.sort(reverse=True)
    result = []
    for f in files[:limit]:
        with open(os.path.join(USER_ANALYSES_DIR, f), "r", encoding="utf-8") as fp:
            result.append(json.load(fp))
    return result

def compress_image(path, max_size=(1920,1080), quality=85):
    img = cv2.imread(path)
    if img is None:
        return path
    h, w = img.shape[:2]
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0]/w, max_size[1]/h)
        nw, nh = int(w*scale), int(h*scale)
        img = cv2.resize(img, (nw, nh))
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    out = path + ".compressed.jpg"
    with open(out, 'wb') as f:
        f.write(buf)
    return out

async def get_weather(lat, lon):
    if not WEATHER_API_KEY:
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"]
                }
    except Exception as e:
        logger.error(f"Ошибка погоды: {e}")
    return None

async def query_llm(prompt, system_message="Ты полезный ассистент. Отвечай на русском, кратко."):
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
        "temperature": 0.7, "max_tokens": 500, "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(LLM_API_URL, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM error: {e}")
    return None

async def query_mllm(prompt, image_base64=None, history=None, system_message="Ты полезный ассистент. Отвечай на русском, кратко."):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if history:
        messages.extend(history)
    user_content = []
    if prompt:
        user_content.append({"type": "text", "text": prompt})
    if image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    messages.append({"role": "user", "content": user_content})
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 1000, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(LLM_API_URL, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"MLLM error: {e}")
    return None

# ================== КЛАВИАТУРЫ ==================
def get_main_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📷 Анализ", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("🤖 AI", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("📊 Данные", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("🛠️ Инструменты", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⚙️ Настройки", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("📞 Контакт", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_back_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_metadata_ask_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("✅ Да", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("⏭️ Нет", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Отмена", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

def get_skip_keyboard(action):
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("⏭️ Пропустить", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("⬅️ Отмена", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

def get_after_photo_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📸 Загрузить ещё", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("⚙️ Настройки", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("🤖 Описать результат", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Меню", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_after_agentic_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📸 Новый анализ", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("💬 Задать вопрос по изображению", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Меню", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_dataset_menu_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📤 Загрузить фото (с разметкой)", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("📤 Загрузить фото (без разметки)", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("📊 Статистика датасета", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("📦 Экспорт датасета", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("❌ Очистить датасет", color=VkKeyboardColor.NEGATIVE)
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_dataset_export_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📦 COCO JSON", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📄 Pascal VOC XML", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📁 YOLO TXT", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_dataset_confirm_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("✅ Сохранить", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("❌ Пропустить", color=VkKeyboardColor.NEGATIVE)
    keyboard.add_line()
    keyboard.add_button("⬅️ Отмена", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_rag_menu_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📤 Загрузить документ", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("📋 Список документов", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("🔍 Поиск по документам", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_settings_keyboard(settings):
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button(f"🎯 Порог: {settings['conf']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"📐 IoU: {settings['iou']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🔢 Макс: {settings['max_det']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🖥️ Устр: {settings['device']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🔄 Сброс", color=VkKeyboardColor.NEGATIVE)
    keyboard.add_button("✅ Готово", color=VkKeyboardColor.POSITIVE)
    return keyboard.get_keyboard()

# ================== RAG МОДУЛЬ (упрощённая версия) ==================
class RAGManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_dir = os.path.join(USER_DOCS_DIR, str(user_id))
        self.index_dir = os.path.join(RAG_INDEX_DIR, str(user_id))
        os.makedirs(self.user_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        self.index = None
        self.chunks = []
        self.sources = []
        self._load()
    def _load(self):
        idx_path = os.path.join(self.index_dir, "index.pkl")
        ch_path = os.path.join(self.index_dir, "chunks.pkl")
        src_path = os.path.join(self.index_dir, "sources.pkl")
        if os.path.exists(idx_path) and os.path.exists(ch_path) and os.path.exists(src_path):
            try:
                with open(idx_path, 'rb') as f: self.index = pickle.load(f)
                with open(ch_path, 'rb') as f: self.chunks = pickle.load(f)
                with open(src_path, 'rb') as f: self.sources = pickle.load(f)
            except: pass
    def _save(self):
        with open(os.path.join(self.index_dir, "index.pkl"), 'wb') as f: pickle.dump(self.index, f)
        with open(os.path.join(self.index_dir, "chunks.pkl"), 'wb') as f: pickle.dump(self.chunks, f)
        with open(os.path.join(self.index_dir, "sources.pkl"), 'wb') as f: pickle.dump(self.sources, f)
    def add_document(self, file_path, filename):
        # упрощённо: только извлечение текста и индексация
        return True
    def search(self, query, top_k=3):
        return []
    def list_documents(self):
        return []
    def delete_document(self, filename):
        pass

# ================== МОДУЛЬ УПРАВЛЕНИЯ ДАТАСЕТАМИ ==================
class DatasetManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.base_dir = os.path.join(USER_DATASETS_DIR, str(user_id))
        self.images_dir = os.path.join(self.base_dir, "images")
        self.annotations_dir = os.path.join(self.base_dir, "annotations")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.base_dir, "metadata.json")
        self.metadata = self._load_metadata()
    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"task": None, "object": None, "images": [], "categories": {}}
    def _save_metadata(self):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    def set_task_and_object(self, task, obj):
        self.metadata["task"] = task
        self.metadata["object"] = obj
        self._save_metadata()
    def add_image(self, image_bytes, detections, width, height, metadata=None, weather=None):
        img_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_fn = f"{img_id}.jpg"
        with open(os.path.join(self.images_dir, img_fn), "wb") as f:
            f.write(image_bytes)
        ann = {"image_id": img_id, "file_name": img_fn, "width": width, "height": height, "detections": detections,
               "metadata": metadata or {}, "weather": weather or {}, "timestamp": datetime.now().isoformat()}
        with open(os.path.join(self.annotations_dir, f"{img_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        self.metadata["images"].append({"id": img_id, "file_name": img_fn, "width": width, "height": height})
        for d in detections:
            cls = d['class']
            if cls not in self.metadata["categories"]:
                self.metadata["categories"][cls] = len(self.metadata["categories"])+1
        self._save_metadata()
        return img_id
    def get_stats(self):
        return {"total_images": len(self.metadata["images"]), "categories": list(self.metadata["categories"].keys()),
                "task": self.metadata.get("task"), "object": self.metadata.get("object")}
    def export_coco(self):
        # упрощённо
        tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        tmp.close()
        return tmp.name
    def export_yolo(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        tmp.close()
        return tmp.name
    def export_voc(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        tmp.close()
        return tmp.name

# ================== АСИНХРОННАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ ==================
async def process_image_vk(user_id, image_path, task, obj, metadata, original_path, settings):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(image_path, 'rb') as f:
                resp = await client.post(
                    "http://127.0.0.1:9000/api/analyze_plant",
                    files={'file': f},
                    data={'task': task, 'object': obj,
                          'conf': settings['conf'], 'iou': settings['iou'],
                          'max_det': settings['max_det'], 'device': settings['device']}
                )
        if resp.status_code != 200:
            send_message(user_id, f"Ошибка сервера: {resp.status_code}")
            return
        data = resp.json()
        if "error" in data:
            send_message(user_id, f"Ошибка: {data['error']}")
            return
        dets = data.get("detections", [])
        counts = data.get("counts", {})
        img_b64 = data["image_base64"]
        img_bytes = base64.b64decode(img_b64)
        annotated_path = save_user_image(user_id, img_bytes, f"annotated_{task}_{obj}")
        report = "Обнаружено:\n" + "\n".join(f"- {cls}: {cnt}" for cls, cnt in counts.items()) if counts else "Объектов не обнаружено."
        send_photo(user_id, annotated_path, caption=report, keyboard=get_after_photo_keyboard())
        os.unlink(annotated_path)
    except Exception as e:
        logger.exception("Ошибка при анализе")
        send_message(user_id, f"Ошибка: {type(e).__name__}")

async def process_agentic_image_vk(user_id, image_path, task, obj, original_path):
    # упрощённая версия агентного анализа
    send_message(user_id, "Агентный анализ временно недоступен. Используйте обычный анализ.")
    return

# ================== ОСНОВНОЙ ЦИКЛ ==================
if __name__ == "__main__":
    logger.info("Бот запущен")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            text = event.text.strip()
            logger.info(f"Сообщение от {user_id}: {text}")
            user = get_user_state(user_id)
            step = user.get("step")

            # Получение вложений
            attachments = []
            if hasattr(event, 'message_id') and event.message_id:
                try:
                    msgs = vk.messages.getById(message_ids=event.message_id)
                    if msgs and msgs.get('count',0)>0:
                        attachments = msgs['items'][0].get('attachments', [])
                except: pass

            # Обработка фото из вложений
            photo_bytes = get_photo_bytes_from_attachments(attachments)
            if photo_bytes:
                tmp_path = f"/tmp/vk_{user_id}_{datetime.now().timestamp()}.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(photo_bytes)
                if step == "PHOTO_WAITING_PHOTO":
                    task = user.get("photo_task")
                    obj = user.get("photo_object")
                    if task and obj:
                        meta = user.get("metadata", {})
                        sett = user.get("settings", DEFAULT_SETTINGS)
                        orig = save_user_image(user_id, photo_bytes, "original")
                        await process_image_vk(user_id, tmp_path, task, obj, meta, orig, sett)
                    os.unlink(tmp_path)
                    continue
                if step == "AGENTIC_WAITING_PHOTO":
                    task = user.get("agentic_task")
                    obj = user.get("agentic_object")
                    if task and obj:
                        orig = save_user_image(user_id, photo_bytes, "agentic_original")
                        await process_agentic_image_vk(user_id, tmp_path, task, obj, orig)
                    os.unlink(tmp_path)
                    continue
                os.unlink(tmp_path)

            # Обработка текстовых команд
            if text == "/start":
                user.clear()
                user["settings"] = DEFAULT_SETTINGS.copy()
                user["step"] = None
                send_message(user_id, "🍏 AI-ассистент садовода (VK версия)\nВыберите действие:", get_main_keyboard())
                continue
            if text == "⬅️ Назад":
                user["step"] = None
                send_message(user_id, "Главное меню", get_main_keyboard())
                continue

            # Главное меню
            if text == "📷 Анализ":
                user["step"] = "ANALYSIS_MENU"
                keyboard = VkKeyboard(inline=True)
                keyboard.add_button("🖼️ Анализ изображений", color=VkKeyboardColor.PRIMARY)
                keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                send_message(user_id, "Выберите тип анализа:", keyboard.get_keyboard())
                continue
            if text == "🖼️ Анализ изображений":
                user["step"] = "PHOTO_SELECT_TASK"
                task_names = list(TASKS.keys())
                keyboard = VkKeyboard(inline=True)
                for name in task_names:
                    keyboard.add_button(name, color=VkKeyboardColor.PRIMARY)
                    keyboard.add_line()
                keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                send_message(user_id, "Выберите задачу:", keyboard.get_keyboard())
                continue
            if step == "PHOTO_SELECT_TASK":
                for name, tid in TASKS.items():
                    if text == name:
                        user["photo_task"] = tid
                        user["step"] = "PHOTO_SELECT_OBJECT"
                        allowed = {"📸 Все классы": "general", "🍎 Плоды яблони": "fruit"}
                        keyboard = VkKeyboard(inline=True)
                        for oname in allowed:
                            keyboard.add_button(oname, color=VkKeyboardColor.PRIMARY)
                            keyboard.add_line()
                        keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                        send_message(user_id, "Выберите объект:", keyboard.get_keyboard())
                        break
                else:
                    send_message(user_id, "Неизвестная задача")
                continue
            if step == "PHOTO_SELECT_OBJECT":
                allowed = {"📸 Все классы": "general", "🍎 Плоды яблони": "fruit"}
                if text in allowed:
                    user["photo_object"] = allowed[text]
                    user["step"] = "ASK_METADATA"
                    send_message(user_id, "Хотите добавить метаданные?", get_metadata_ask_keyboard())
                    continue
            if step == "ASK_METADATA":
                if text == "✅ Да":
                    user["metadata"] = {}
                    user["step"] = "SORT"
                    send_message(user_id, "🌱 Введите сорт (или пропуск):", get_skip_keyboard("sort"))
                elif text == "⏭️ Нет":
                    user["metadata"] = {}
                    user["step"] = "PHOTO_WAITING_PHOTO"
                    send_message(user_id, "✅ Отправьте фото.")
                elif text == "⬅️ Отмена":
                    user.clear()
                    user["settings"] = DEFAULT_SETTINGS.copy()
                    send_message(user_id, "Отменено", get_main_keyboard())
                continue
            if step == "SORT":
                if text == "⏭️ Пропустить":
                    user["metadata"]["sort"] = None
                else:
                    user["metadata"]["sort"] = text
                user["step"] = "PHASE"
                send_message(user_id, "🌸 Введите фазу (или пропуск):", get_skip_keyboard("phase"))
                continue
            if step == "PHASE":
                if text == "⏭️ Пропустить":
                    user["metadata"]["phase"] = None
                else:
                    user["metadata"]["phase"] = text
                user["step"] = "LOCATION"
                send_message(user_id, "📍 Введите геолокацию (широта,долгота) или пропуск:", get_skip_keyboard("location"))
                continue
            if step == "LOCATION":
                if text == "⏭️ Пропустить":
                    user["metadata"]["latitude"] = None
                    user["metadata"]["longitude"] = None
                else:
                    try:
                        lat, lon = map(float, text.replace(',',' ').split())
                        user["metadata"]["latitude"] = lat
                        user["metadata"]["longitude"] = lon
                    except:
                        send_message(user_id, "Неверный формат, пропускаем.")
                user["step"] = "COMMENT"
                send_message(user_id, "📝 Введите комментарий (или пропуск):", get_skip_keyboard("comment"))
                continue
            if step == "COMMENT":
                if text == "⏭️ Пропустить":
                    user["metadata"]["comment"] = None
                else:
                    user["metadata"]["comment"] = text
                user["step"] = "PHOTO_WAITING_PHOTO"
                send_message(user_id, "✅ Метаданные сохранены. Отправьте фото.")
                continue
            # Остальные команды (настройки, статистика и т.д.) опущены для краткости
            send_message(user_id, "Нажмите /start", get_main_keyboard())
