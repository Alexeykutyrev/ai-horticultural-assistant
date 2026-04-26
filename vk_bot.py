#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
from vk_api.upload import VkUpload

import logging
import httpx
import time
import shutil
import os
import json
import base64
import csv
import io
import cv2
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import tempfile
import pickle
import re
import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import ssl
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

# Для RAG
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import PyPDF2
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ================== КОНФИГУРАЦИЯ (ЗАМЕНИТЕ НА СВОИ ЗНАЧЕНИЯ В .env) ==================
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-vl-7b-instruct")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "ЗАМЕНИТЕ_НА_КЛЮЧ_OPENWEATHERMAP")

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

SUBSCRIBERS_FILE = "subscribers.json"

def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_subscribers(subscribers):
    with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(subscribers), f, ensure_ascii=False, indent=2)

subscribers = load_subscribers()
ADMIN_ID = 541120018
ADMIN_USERNAME = "alexeykutyrev"

TASKS = {
    "🔍 Распознавание объектов (Obj Det)": "detection",
    "✂️ Сегментация экземпляров (Inst Seg)": "instance_seg",
    "🧩 Семантическая сегментация (Sem Seg)": "semantic_seg",
    "🏷️ Классификация (Classif)": "classification",
    "🔑 Распознавание точек (Keypoint)": "keypoint",
    "🖼️ Мультимодальный анализ (Multimodal)": "multimodal"
}

def get_allowed_objects(task: str) -> Dict[str, str]:
    if task == "detection":
        return {
            "📸 Все классы": "general",
            "🍎 Плоды яблони": "fruit",
            "🌸 Цветки яблони": "apple_flower",
            "🍏 Завязи яблони": "apple_ovary",
            "🍂 Болезни листьев": "disease",
            "🍎 Плоды яблони (ночь)": "apple_night",
            "🌳 Деревья-опылители": "apple_pollinator_tree",
            "🌱 Почки яблони": "apple_buds",
            "📏 Инвентаризация": "apple_tree_trunk",
            "🍇 Виноград": "grape",
            "🪰 Плодовые мушки": "fruit_flies",
            "🍓 Цветки земляники": "strawberry_flower"
        }
    elif task == "instance_seg":
        return {
            "📸 Все классы": "general",
            "🌸 Цветки яблони": "apple_flower",
            "🍎 Плоды яблони": "apple_fruit",
            "🍎 Сортировка плодов": "apple_sort",
            "🔬 Микроскоп (парша/роса)": "microscope_scab_powdery_mildew"
        }
    elif task == "semantic_seg":
        return {"📸 Все классы": "general"}
    elif task == "classification":
        return {"📸 Все классы": "general"}
    elif task == "keypoint":
        return {
            "👤 Люди": "general",
            "🌳 Деревья яблони": "apple_tree_pose"
        }
    else:
        return {"📸 Все классы": "general"}

PHASES = {
    "🌱 Всходы": "seedling",
    "🌿 Вегетация": "vegetation",
    "🌸 Цветение": "flowering",
    "🍎 Плодоношение": "fruiting",
    "🍂 Созревание": "ripening",
    "🌾 Уборка": "harvest"
}

DEFAULT_SETTINGS = {
    "conf": 0.15,
    "iou": 0.45,
    "max_det": 300,
    "device": "cpu",
    "video_fps": 60,
    "video_max_frames": 500,
    "return_video": True,
    "track": True,
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "match_thresh": 0.8
}

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================

def save_user_image(user_id: int, image_bytes: bytes, suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(USER_IMAGES_DIR, f"{user_id}_{ts}_{suffix}.jpg")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path

def save_analysis(user_id: int, data: Dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(USER_ANALYSES_DIR, f"{user_id}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def get_user_analyses(user_id: int, limit: int = 10) -> List[Dict]:
    files = [f for f in os.listdir(USER_ANALYSES_DIR) if f.startswith(f"{user_id}_") and f.endswith(".json")]
    files.sort(reverse=True)
    result = []
    for f in files[:limit]:
        with open(os.path.join(USER_ANALYSES_DIR, f), "r", encoding="utf-8") as fp:
            result.append(json.load(fp))
    return result

def export_to_csv(user_id: int) -> str:
    analyses = get_user_analyses(user_id, 1000)
    if not analyses:
        return None
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["timestamp", "task", "object", "sort", "phase", "lat", "lon", "comment",
                "temp", "humidity", "weather_desc", "total", "detections_json", "image_path", "video_annotated"])
    for a in analyses:
        w.writerow([
            a.get("timestamp"),
            a.get("task"),
            a.get("object"),
            a.get("metadata", {}).get("sort"),
            a.get("metadata", {}).get("phase"),
            a.get("metadata", {}).get("latitude"),
            a.get("metadata", {}).get("longitude"),
            a.get("metadata", {}).get("comment"),
            a.get("weather", {}).get("temp"),
            a.get("weather", {}).get("humidity"),
            a.get("weather", {}).get("description"),
            len(a.get("detections", [])),
            json.dumps(a.get("detections", []), ensure_ascii=False),
            a.get("image_path"),
            a.get("video_annotated", "")
        ])
    csv_path = os.path.join(USER_ANALYSES_DIR, f"{user_id}_export.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(out.getvalue())
    return csv_path

async def generate_stats(user_id: int) -> Optional[str]:
    analyses = get_user_analyses(user_id, 1000)
    if not analyses:
        return None
    by_date = defaultdict(int)
    for a in analyses:
        if a.get("timestamp"):
            by_date[a["timestamp"][:10]] += 1
    dates = sorted(by_date.keys())
    counts = [by_date[d] for d in dates]

    class_counts = defaultdict(int)
    class_confidences = defaultdict(list)
    for a in analyses[-200:]:
        for d in a.get("detections", []):
            cls = d.get("class", "unknown")
            class_counts[cls] += 1
            if "confidence" in d:
                class_confidences[cls].append(d["confidence"])

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(dates, counts, color='green', alpha=0.7)
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Количество анализов')
    ax1.set_title('Динамика анализов')
    ax1.tick_params(axis='x', rotation=45)

    if sorted_classes:
        classes = [c for c, _ in sorted_classes]
        values = [v for _, v in sorted_classes]
        avg_confs = []
        for cls in classes:
            if cls in class_confidences and class_confidences[cls]:
                avg_confs.append(sum(class_confidences[cls]) / len(class_confidences[cls]))
            else:
                avg_confs.append(0)

        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, values, color='skyblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Количество обнаружений')
        ax2.set_title('Топ-10 классов (последние 200 анализов)')

        for i, (v, conf) in enumerate(zip(values, avg_confs)):
            ax2.text(v + max(values)*0.01, i, f'{conf:.2f}', va='center')
    else:
        ax2.text(0.5, 0.5, 'Нет данных об объектах', ha='center', va='center')
        ax2.set_title('Топ-10 классов')

    plt.tight_layout()
    tmp = f"/tmp/stats_{user_id}_{datetime.now().timestamp()}.png"
    plt.savefig(tmp)
    plt.close()
    return tmp

async def generate_dataset_progress(user_id: int) -> Optional[str]:
    dm = DatasetManager(user_id)
    stats = dm.get_stats()
    if stats["total_images"] == 0:
        return None
    total = stats["total_images"]
    target = 1000
    percent = min(100, int(total / target * 100))

    class_counts = {}
    for img_info in dm.metadata["images"]:
        ann_path = os.path.join(dm.annotations_dir, f"{img_info['id']}.json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            for det in ann_data["detections"]:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh([0], [percent], color='green', height=0.4)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel('Процент выполнения')
    ax1.set_title(f'Прогресс сбора датасета: {total}/{target} ({percent}%)')
    ax1.text(percent/2, 0, f'{percent}%', ha='center', va='center', color='white', fontweight='bold')

    if class_counts:
        classes = list(class_counts.keys())
        values = list(class_counts.values())
        ax2.bar(classes, values, color='orange')
        ax2.set_xlabel('Класс')
        ax2.set_ylabel('Количество')
        ax2.set_title('Количество изображений по классам')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        ax2.set_title('Распределение по классам')

    plt.tight_layout()
    tmp = f"/tmp/dataset_progress_{user_id}_{datetime.now().timestamp()}.png"
    plt.savefig(tmp)
    plt.close()
    return tmp

async def generate_extremes_report(user_id: int) -> str:
    analyses = get_user_analyses(user_id, 1000)
    if not analyses:
        return "📭 Нет данных для анализа."

    max_objects = 0
    max_objects_analysis = None
    max_avg_confidence = 0.0
    max_avg_confidence_analysis = None
    class_total_counts = defaultdict(int)
    max_width = 0
    max_height = 0
    max_area_analysis = None
    total_detections = 0
    max_confidence_value = 0.0
    min_confidence_value = 1.0
    earliest_date = None
    latest_date = None

    for a in analyses:
        dets = a.get("detections", [])
        cnt = len(dets)
        total_detections += cnt
        if cnt > max_objects:
            max_objects = cnt
            max_objects_analysis = a

        confs = [d.get("confidence", 0) for d in dets if d.get("confidence")]
        if confs:
            avg_conf = sum(confs) / len(confs)
            if avg_conf > max_avg_confidence:
                max_avg_confidence = avg_conf
                max_avg_confidence_analysis = a
            max_conf = max(confs)
            min_conf = min(confs)
            if max_conf > max_confidence_value:
                max_confidence_value = max_conf
            if min_conf < min_confidence_value:
                min_confidence_value = min_conf

        for d in dets:
            cls = d.get("class", "unknown")
            class_total_counts[cls] += 1

        w = a.get("image_width", 0)
        h = a.get("image_height", 0)
        if w * h > max_width * max_height:
            max_width = w
            max_height = h
            max_area_analysis = a

        ts = a.get("timestamp")
        if ts:
            dt = datetime.fromisoformat(ts)
            if earliest_date is None or dt < earliest_date:
                earliest_date = dt
            if latest_date is None or dt > latest_date:
                latest_date = dt

    if class_total_counts:
        rarest_class = min(class_total_counts.items(), key=lambda x: x[1])
        most_common_class = max(class_total_counts.items(), key=lambda x: x[1])
    else:
        rarest_class = ("нет", 0)
        most_common_class = ("нет", 0)

    report = "🏆 **Статистика экстремумов**\n\n"
    report += f"📅 Период: с {earliest_date.date() if earliest_date else '—'} по {latest_date.date() if latest_date else '—'}\n"
    report += f"📸 Всего анализов: {len(analyses)}\n"
    report += f"🔍 Всего обнаружений: {total_detections}\n\n"

    report += f"**Максимум объектов на одном изображении:** {max_objects}\n"
    if max_objects_analysis:
        ts = max_objects_analysis.get("timestamp", "?")[:16]
        report += f"   └ Дата: {ts}\n"

    report += f"\n**Максимальная средняя уверенность в анализе:** {max_avg_confidence:.3f}\n"
    if max_avg_confidence_analysis:
        ts = max_avg_confidence_analysis.get("timestamp", "?")[:16]
        report += f"   └ Дата: {ts}\n"

    report += f"\n**Самая высокая уверенность детекции:** {max_confidence_value:.3f}\n"
    report += f"**Самая низкая уверенность детекции:** {min_confidence_value:.3f}\n"

    report += f"\n**Самый частый класс:** {most_common_class[0]} – {most_common_class[1]} раз\n"
    report += f"**Самый редкий класс:** {rarest_class[0]} – {rarest_class[1]} раз\n"

    report += f"\n**Самое большое изображение:** {max_width} x {max_height} пикс.\n"
    if max_area_analysis:
        ts = max_area_analysis.get("timestamp", "?")[:16]
        report += f"   └ Дата: {ts}\n"

    return report

async def get_weather(lat: float, lon: float) -> Optional[Dict]:
    if not WEATHER_API_KEY or WEATHER_API_KEY == "ЗАМЕНИТЕ_НА_КЛЮЧ_OPENWEATHERMAP":
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"],
                    "rain": data.get("rain", {}).get("1h", 0),
                    "clouds": data["clouds"]["all"]
                }
    except Exception as e:
        logging.error(f"Ошибка погоды: {e}")
    return None

async def query_llm(prompt: str,
                    system_message: str = "Ты полезный ассистент. Отвечай только на русском языке, кратко и по делу.") -> Optional[str]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(LLM_API_URL, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"LLM error: {e}")
    return None

async def query_mllm(prompt: str, image_base64: str = None, history: list = None,
                     system_message: str = "Ты полезный ассистент. Отвечай только на русском языке, кратко и по делу.") -> Optional[str]:
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
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(LLM_API_URL, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"MLLM error: {e}")
    return None

def compress_image(path, max_size=(1920, 1080), quality=85):
    img = cv2.imread(path)
    if img is None:
        return path
    h, w = img.shape[:2]
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0] / w, max_size[1] / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    out = path + ".compressed.jpg"
    with open(out, 'wb') as f:
        f.write(buf)
    return out

# ================== ФУНКЦИИ ЭКСПОРТА ==================
def generate_coco_json(detections: List[Dict], image_path: str, image_width: int, image_height: int, task: str,
                       object_name: str) -> str:
    categories = {}
    annotations = []
    for i, det in enumerate(detections):
        cls_name = det['class']
        if cls_name not in categories:
            categories[cls_name] = len(categories) + 1
        cat_id = categories[cls_name]
        bbox = det.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        annotation = {
            "id": i + 1,
            "image_id": 1,
            "category_id": cat_id,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
            "confidence": det.get('confidence', 1.0)
        }
        annotations.append(annotation)

    categories_list = [{"id": v, "name": k} for k, v in categories.items()]
    info = {
        "year": datetime.now().year,
        "version": "1.0",
        "description": f"PlantVisionAI export - {task}/{object_name}",
        "date_created": datetime.now().isoformat()
    }
    images = [{
        "id": 1,
        "file_name": os.path.basename(image_path),
        "width": image_width,
        "height": image_height
    }]
    coco_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(coco_dict, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name

def generate_pascal_voc_xml(detections: List[Dict], image_path: str, image_width: int, image_height: int, task: str,
                            object_name: str) -> str:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "PlantVisionAI"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "PlantVisionAI Export"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det['class']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = det.get('bbox')
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x1)
            ET.SubElement(bndbox, "ymin").text = str(y1)
            ET.SubElement(bndbox, "xmax").text = str(x2)
            ET.SubElement(bndbox, "ymax").text = str(y2)

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8')
    tmp.write(xml_str)
    tmp.close()
    return tmp.name

def generate_yolo_txt(detections: List[Dict], image_width: int, image_height: int, class_to_id: Dict[str, int]) -> str:
    lines = []
    for det in detections:
        cls_name = det['class']
        if cls_name not in class_to_id:
            continue
        cls_id = class_to_id[cls_name]
        bbox = det.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name

# ================== RAG МОДУЛЬ ==================
class RAGManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.user_docs_dir = os.path.join(USER_DOCS_DIR, str(user_id))
        self.user_index_dir = os.path.join(RAG_INDEX_DIR, str(user_id))
        os.makedirs(self.user_docs_dir, exist_ok=True)
        os.makedirs(self.user_index_dir, exist_ok=True)
        self.index_path = os.path.join(self.user_index_dir, "index.pkl")
        self.chunks_path = os.path.join(self.user_index_dir, "chunks.pkl")
        self.sources_path = os.path.join(self.user_index_dir, "sources.pkl")
        self.model = None
        self.index = None
        self.chunks = []
        self.sources = []
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path) and os.path.exists(self.sources_path):
            try:
                with open(self.index_path, 'rb') as f:
                    self.index = pickle.load(f)
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                with open(self.sources_path, 'rb') as f:
                    self.sources = pickle.load(f)
            except:
                self.index = None
                self.chunks = []
                self.sources = []

    def _save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(self.sources_path, 'wb') as f:
            pickle.dump(self.sources, f)

    def _get_model(self):
        if self.model is None and RAG_AVAILABLE:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self.model

    def _save_document(self, file_path: str, filename: str) -> Optional[str]:
        safe_filename = os.path.basename(filename)
        dest_path = os.path.join(self.user_docs_dir, safe_filename)
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(safe_filename)
            counter = 1
            while os.path.exists(os.path.join(self.user_docs_dir, f"{base}_{counter}{ext}")):
                counter += 1
            safe_filename = f"{base}_{counter}{ext}"
            dest_path = os.path.join(self.user_docs_dir, safe_filename)
        try:
            shutil.copy2(file_path, dest_path)
            return safe_filename
        except Exception as e:
            logging.error(f"Ошибка сохранения документа: {e}")
            return None

    def add_document(self, file_path: str, filename: str) -> bool:
        text = ""
        ext = filename.lower().split('.')[-1]

        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext == 'pdf':
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logging.error(f"Ошибка извлечения PDF: {e}")
                return False
        elif ext == 'docx':
            if not DOCX_AVAILABLE:
                return False
            try:
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    if para.text:
                        text += para.text + "\n"
            except Exception as e:
                logging.error(f"Ошибка извлечения DOCX: {e}")
                return False
        else:
            return False

        if not text.strip():
            return False

        sentences = re.split(r'(?<=[.!?])\s+', text)
        new_chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < 500:
                current_chunk += " " + sent
            else:
                if current_chunk:
                    new_chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk:
            new_chunks.append(current_chunk.strip())

        if not new_chunks:
            return False

        model = self._get_model()
        if model is None:
            return False

        embeddings = model.encode(new_chunks, convert_to_numpy=True)
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.sources = []
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.sources.extend([filename] * len(new_chunks))
        self._save_index()

        saved_name = self._save_document(file_path, filename)
        if saved_name is None:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.chunks_path):
                os.remove(self.chunks_path)
            if os.path.exists(self.sources_path):
                os.remove(self.sources_path)
            self.index = None
            self.chunks = []
            self.sources = []
            return False
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if self.index is None or not self.chunks:
            return []
        model = self._get_model()
        if model is None:
            return []
        query_emb = model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append({
                    "text": self.chunks[i],
                    "source": self.sources[i] if i < len(self.sources) else "неизвестно"
                })
        return results

    def list_documents(self) -> List[str]:
        return [f for f in os.listdir(self.user_docs_dir) if os.path.isfile(os.path.join(self.user_docs_dir, f))]

    def delete_document(self, filename: str):
        file_path = os.path.join(self.user_docs_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.chunks_path):
            os.remove(self.chunks_path)
        if os.path.exists(self.sources_path):
            os.remove(self.sources_path)
        self.index = None
        self.chunks = []
        self.sources = []
        for f in self.list_documents():
            self.add_document(os.path.join(self.user_docs_dir, f), f)
        return True

# ================== МОДУЛЬ УПРАВЛЕНИЯ ДАТАСЕТАМИ ==================
class DatasetManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.user_dataset_dir = os.path.join(USER_DATASETS_DIR, str(user_id))
        self.images_dir = os.path.join(self.user_dataset_dir, "images")
        self.annotations_dir = os.path.join(self.user_dataset_dir, "annotations")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.user_dataset_dir, "metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"task": None, "object": None, "images": [], "categories": {}}

    def _save_metadata(self):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def set_task_and_object(self, task: str, obj: str):
        self.metadata["task"] = task
        self.metadata["object"] = obj
        self._save_metadata()

    def add_image(self, image_bytes: bytes, detections: List[Dict], image_width: int, image_height: int,
                  metadata: Dict = None, weather: Dict = None) -> str:
        img_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        ann = {
            "image_id": img_id,
            "file_name": img_filename,
            "width": image_width,
            "height": image_height,
            "detections": detections,
            "metadata": metadata if metadata else {},
            "weather": weather if weather else {},
            "timestamp": datetime.now().isoformat()
        }
        ann_path = os.path.join(self.annotations_dir, f"{img_id}.json")
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

        self.metadata["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": image_width,
            "height": image_height,
            "timestamp": datetime.now().isoformat()
        })
        for d in detections:
            cls = d['class']
            if cls not in self.metadata["categories"]:
                self.metadata["categories"][cls] = len(self.metadata["categories"]) + 1
        self._save_metadata()
        return img_id

    def get_stats(self) -> Dict:
        return {
            "total_images": len(self.metadata["images"]),
            "categories": list(self.metadata["categories"].keys()),
            "task": self.metadata.get("task"),
            "object": self.metadata.get("object"),
            "last_images": self.metadata["images"][-5:] if self.metadata["images"] else []
        }

    def export_coco(self) -> str:
        categories = [{"id": v, "name": k} for k, v in self.metadata["categories"].items()]
        images = []
        annotations = []
        ann_id = 1
        for img_info in self.metadata["images"]:
            img_id = img_info["id"]
            images.append({
                "id": img_id,
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"]
            })
            ann_path = os.path.join(self.annotations_dir, f"{img_info['id']}.json")
            if os.path.exists(ann_path):
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                for det in ann_data["detections"]:
                    cls = det['class']
                    if cls not in self.metadata["categories"]:
                        continue
                    cat_id = self.metadata["categories"][cls]
                    bbox = det.get('bbox')
                    if not bbox or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "confidence": det.get('confidence', 1.0)
                    })
                    ann_id += 1
        coco_dict = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"PlantVisionAI dataset for task {self.metadata['task']} object {self.metadata['object']}",
                "date_created": datetime.now().isoformat()
            },
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(coco_dict, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return tmp.name

    def export_yolo(self) -> str:
        tmp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(tmp_dir, "images")
        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        for img_info in self.metadata["images"]:
            src_img = os.path.join(self.images_dir, img_info["file_name"])
            dst_img = os.path.join(images_dir, img_info["file_name"])
            shutil.copy2(src_img, dst_img)

            ann_path = os.path.join(self.annotations_dir, f"{img_info['id']}.json")
            if os.path.exists(ann_path):
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                lines = []
                for det in ann_data["detections"]:
                    cls = det['class']
                    if cls not in self.metadata["categories"]:
                        continue
                    cls_id = self.metadata["categories"][cls] - 1
                    bbox = det.get('bbox')
                    if not bbox or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    img_w = img_info["width"]
                    img_h = img_info["height"]
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                if lines:
                    txt_path = os.path.join(labels_dir, os.path.splitext(img_info["file_name"])[0] + ".txt")
                    with open(txt_path, 'w') as f:
                        f.write("\n".join(lines))

        classes = [k for k, v in sorted(self.metadata["categories"].items(), key=lambda x: x[1])]
        classes_path = os.path.join(tmp_dir, "classes.txt")
        with open(classes_path, 'w') as f:
            f.write("\n".join(classes))

        zip_path = os.path.join(USER_DATASETS_DIR, f"{self.user_id}_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), tmp_dir))
        shutil.rmtree(tmp_dir)
        return zip_path

    def export_voc(self) -> str:
        tmp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(tmp_dir, "images")
        annotations_dir = os.path.join(tmp_dir, "annotations")
        os.makedirs(images_dir)
        os.makedirs(annotations_dir)

        for img_info in self.metadata["images"]:
            src_img = os.path.join(self.images_dir, img_info["file_name"])
            dst_img = os.path.join(images_dir, img_info["file_name"])
            shutil.copy2(src_img, dst_img)

            ann_path = os.path.join(self.annotations_dir, f"{img_info['id']}.json")
            if os.path.exists(ann_path):
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "folder").text = "PlantVisionAI"
                ET.SubElement(annotation, "filename").text = img_info["file_name"]
                source = ET.SubElement(annotation, "source")
                ET.SubElement(source, "database").text = "PlantVisionAI Dataset"
                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = str(img_info["width"])
                ET.SubElement(size, "height").text = str(img_info["height"])
                ET.SubElement(size, "depth").text = "3"
                ET.SubElement(annotation, "segmented").text = "0"
                for det in ann_data["detections"]:
                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = det['class']
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    bbox = det.get('bbox')
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        bndbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = str(x1)
                        ET.SubElement(bndbox, "ymin").text = str(y1)
                        ET.SubElement(bndbox, "xmax").text = str(x2)
                        ET.SubElement(bndbox, "ymax").text = str(y2)
                xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
                xml_path = os.path.join(annotations_dir, os.path.splitext(img_info["file_name"])[0] + ".xml")
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(xml_str)

        zip_path = os.path.join(USER_DATASETS_DIR, f"{self.user_id}_voc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), tmp_dir))
        shutil.rmtree(tmp_dir)
        return zip_path

# ================== НАСТРОЙКА ЛОГГЕРА ==================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ================== VK БОТ (vk_api) ==================
VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN", "ЗАМЕНИТЕ_НА_ТОКЕН_ГРУППЫ")
GROUP_ID = 147618176

vk_session = vk_api.VkApi(token=VK_GROUP_TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
upload = VkUpload(vk_session)

user_data = {}

def get_user_state(user_id):
    if user_id not in user_data:
        user_data[user_id] = {"settings": DEFAULT_SETTINGS.copy(), "step": None}
    return user_data[user_id]

# Состояния
ASK_METADATA = 0
SORT = 1
PHASE = 2
LOCATION = 3
COMMENT = 4
SETTINGS_CONF = 5
SETTINGS_IOU = 6
SETTINGS_MAXDET = 7
SETTINGS_DEVICE = 8
WAITING_MODEL_FILE = 9
WAITING_MODEL_SELECTION = 10
WAITING_TEST_PHOTO = 11
LLM_CHAT_MODE = 12
ASK_ABOUT_RESULT = 13
VIDEO_SELECT_TASK = 14
VIDEO_SELECT_OBJECT = 15
VIDEO_READY = 16
WAITING_VIDEO = 17
TRACKER_SETTINGS = 18
EXPORT_FORMAT_SELECTION = 19
AGENTIC_SELECT_TASK = 20
AGENTIC_SELECT_OBJECT = 21
AGENTIC_WAITING_PHOTO = 22
AGENTIC_DIALOG = 23
RAG_MENU = 24
RAG_WAITING_DOCUMENT = 25
RAG_DELETE_CONFIRM = 26
RAG_SEARCH_MODE = 27
PHOTO_SELECT_TASK = 28
PHOTO_SELECT_OBJECT = 29
PHOTO_WAITING_PHOTO = 30
DATASET_MENU = 31
DATASET_SELECT_TASK = 32
DATASET_SELECT_OBJECT = 33
DATASET_WAITING_PHOTOS = 34
DATASET_CONFIRM_ANNOTATION = 35
DATASET_EXPORT_MENU = 36
DATASET_WAITING_PHOTOS_NO_ANNOT = 37

ANALYSIS_MENU = 100
AI_MENU = 101
DATA_MENU = 102
TOOLS_MENU = 103

# ================== МНОГОУРОВНЕВЫЕ КЛАВИАТУРЫ ==================
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

def get_analysis_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("🖼️ Анализ изображений", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("🎥 Анализ видео", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_ai_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("💬 Чат с LLM", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("🤖 Agentic AI", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_data_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📁 Сбор датасета", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("📚 База знаний", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_tools_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("🧪 Тест модели", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("📊 Статистика", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📋 История", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("📥 Экспорт CSV", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
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
    keyboard.add_button("🔄 Повторить", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🤖 Описать результат", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("💬 Задать вопрос", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🔍 Задать вопрос с RAG", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("📥 Экспорт", color=VkKeyboardColor.SECONDARY)
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

def get_video_ready_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("⚙️ Настройки видео", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📤 Отправить видео", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_video_settings_keyboard(settings):
    result_text = "Видео" if settings.get('return_video', False) else "Коллаж"
    track_text = "Вкл" if settings.get('track', False) else "Выкл"
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button(f"🎯 Порог: {settings['conf']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button(f"📐 IoU: {settings['iou']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🖥️ Устр: {settings['device']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button(f"🎞️ FPS: {settings['video_fps']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"📽️ Кадры: {settings['video_max_frames']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button(f"📹 Результат: {result_text}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🔄 Трекинг: {track_text}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("⚙️ Параметры трекера", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()

def get_tracker_settings_keyboard(settings):
    type_text = "ByteTrack" if settings['tracker_type'] == "bytetrack" else "BotSORT"
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button(f"📌 Тип: {type_text}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🔺 High: {settings['track_high_thresh']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button(f"🔻 Low: {settings['track_low_thresh']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button(f"🆕 New: {settings['new_track_thresh']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button(f"🤝 Match: {settings['match_thresh']}", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🔄 Сброс", color=VkKeyboardColor.NEGATIVE)
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

def get_export_format_keyboard():
    keyboard = VkKeyboard(inline=True)
    keyboard.add_button("📦 COCO JSON", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📄 Pascal VOC XML", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("📁 YOLO TXT", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🔙 Отмена", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

# ================== ФУНКЦИИ ОТПРАВКИ ==================
def send_message(user_id, message, keyboard=None):
    params = {"user_id": user_id, "message": message, "random_id": get_random_id()}
    if keyboard:
        params["keyboard"] = keyboard
    vk.messages.send(**params)

def send_photo(user_id, photo_path, caption="", keyboard=None):
    try:
        photo = upload.photo_messages(photo_path)[0]
        attachment = f"photo{photo['owner_id']}_{photo['id']}"
        params = {"user_id": user_id, "attachment": attachment, "message": caption, "random_id": get_random_id()}
        if keyboard:
            params["keyboard"] = keyboard
        vk.messages.send(**params)
    except Exception as e:
        logger.exception("Ошибка отправки фото")
        send_message(user_id, "❌ Не удалось отправить изображение.")

def send_document(user_id, file_path, title="file"):
    try:
        doc = upload.document_message(file_path, title=title, peer_id=user_id)
        attachment = f"doc{doc[0]['owner_id']}_{doc[0]['id']}"
        vk.messages.send(user_id=user_id, attachment=attachment, random_id=get_random_id())
    except Exception as e:
        logger.exception("Ошибка отправки документа")
        send_message(user_id, "❌ Не удалось отправить документ.")

def send_location(user_id, lat, lon):
    send_message(user_id, f"📍 Местоположение: https://maps.google.com/?q={lat},{lon}")

def get_photo_bytes_from_attachments(attachments):
    for att in attachments:
        if att.get('type') == 'photo':
            photo = att.get('photo', {})
            sizes = photo.get('sizes', [])
            if not sizes:
                continue
            max_size = max(sizes, key=lambda s: s.get('height', 0) * s.get('width', 0))
            photo_url = max_size.get('url')
            if not photo_url:
                continue
            response = vk_session.http.get(photo_url)
            return response.content
    return None

# ================== АСИНХРОННЫЕ ФУНКЦИИ ОБРАБОТКИ ==================
async def process_image_vk(user_id, image_path, task, obj, metadata, original_path, settings):
    annotated_path = None
    compressed = None
    weather = None
    if metadata.get("latitude") and metadata.get("longitude"):
        weather = await get_weather(metadata["latitude"], metadata["longitude"])
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(image_path, 'rb') as f:
                resp = await client.post(
                    "http://127.0.0.1:9000/api/analyze_plant",
                    files={'file': f},
                    data={
                        'task': task, 'object': obj,
                        'conf': settings['conf'], 'iou': settings['iou'],
                        'max_det': settings['max_det'], 'device': settings['device']
                    }
                )
        if resp.status_code != 200:
            send_message(user_id, f"❌ Ошибка сервера: {resp.status_code}")
            return
        data = resp.json()
        if "error" in data:
            send_message(user_id, f"❌ {data['error']}")
            return
        dets = data.get("detections", [])
        counts = data.get("counts", {})
        img_b64 = data["image_base64"]
        img_bytes = base64.b64decode(img_b64)
        annotated_path = save_user_image(user_id, img_bytes, f"vk_annotated_{task}_{obj}")
        img_width = data.get("image_width", 640)
        img_height = data.get("image_height", 640)

        if not dets:
            report = "✅ Объектов не обнаружено."
        else:
            by_cls = defaultdict(list)
            for d in dets:
                by_cls[d['class']].append(d['confidence'])
            lines = ["🔎 **Обнаружено:**"]
            for cls, confs in by_cls.items():
                lines.append(f"- {cls}: {len(confs)} (ср. {sum(confs)/len(confs):.2f})")
            all_conf = [d['confidence'] for d in dets]
            lines.append(f"\n📊 Ср. уверенность: {sum(all_conf)/len(all_conf):.2f}")
            report = "\n".join(lines)
        if weather:
            report += f"\n\n🌦️ {weather['temp']}°C, {weather['description']}"
        report += f"\n⚙️ conf={settings['conf']}, iou={settings['iou']}, max_det={settings['max_det']}, device={settings['device']}"

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "task": task,
            "object": obj,
            "metadata": metadata,
            "weather": weather,
            "detections": dets,
            "counts": counts,
            "image_path": original_path,
            "annotated_path": annotated_path,
            "image_width": img_width,
            "image_height": img_height,
            "settings": settings
        }
        save_analysis(user_id, analysis)

        user = get_user_state(user_id)
        user['last_detections'] = dets
        user['last_counts'] = counts
        user['last_metadata'] = metadata
        user['last_weather'] = weather
        user['last_original_path'] = original_path
        user['last_task'] = task
        user['last_object'] = obj
        user['last_image_width'] = img_width
        user['last_image_height'] = img_height
        user['analysis_context'] = analysis
        user['step'] = None

        compressed = compress_image(annotated_path, max_size=(1920, 1080), quality=85)
        send_photo(user_id, compressed, caption=report, keyboard=get_after_photo_keyboard())

        if metadata.get("latitude") and metadata.get("longitude"):
            send_location(user_id, metadata["latitude"], metadata["longitude"])

    except Exception as e:
        logger.exception("Ошибка")
        send_message(user_id, f"❌ Ошибка: {type(e).__name__}")
    finally:
        if compressed and os.path.exists(compressed):
            os.remove(compressed)
        if annotated_path and os.path.exists(annotated_path):
            os.remove(annotated_path)

async def process_agentic_image_vk(user_id, image_path, task, obj, original_path):
    settings = get_user_state(user_id)["settings"]
    annotated_yolo_path = None
    annotated_mllm_path = None
    compressed_yolo = None
    compressed_mllm = None
    weather = None
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(image_path, 'rb') as f:
                resp = await client.post(
                    "http://127.0.0.1:9000/api/analyze_plant",
                    files={'file': f},
                    data={
                        'task': task, 'object': obj,
                        'conf': settings['conf'], 'iou': settings['iou'],
                        'max_det': settings['max_det'], 'device': settings['device']
                    }
                )
        if resp.status_code != 200:
            send_message(user_id, f"❌ Ошибка сервера: {resp.status_code}")
            return
        data = resp.json()
        if "error" in data:
            send_message(user_id, f"❌ {data['error']}")
            return
        dets = data.get("detections", [])
        counts = data.get("counts", {})
        img_b64 = data["image_base64"]
        img_bytes_yolo = base64.b64decode(img_b64)
        annotated_yolo_path = save_user_image(user_id, img_bytes_yolo, f"agentic_yolo_{task}_{obj}")
        img_width = data.get("image_width", 640)
        img_height = data.get("image_height", 640)

        if not dets:
            yolo_report = "✅ Объектов не обнаружено."
        else:
            by_cls = defaultdict(list)
            for d in dets:
                by_cls[d['class']].append(d['confidence'])
            lines = ["🔎 **Обнаружено:**"]
            for cls, confs in by_cls.items():
                lines.append(f"- {cls}: {len(confs)} (ср. {sum(confs)/len(confs):.2f})")
            all_conf = [d['confidence'] for d in dets]
            lines.append(f"\n📊 Ср. уверенность: {sum(all_conf)/len(all_conf):.2f}")
            yolo_report = "\n".join(lines)
        yolo_report += f"\n⚙️ conf={settings['conf']}, iou={settings['iou']}, max_det={settings['max_det']}, device={settings['device']}"

        compressed_yolo = compress_image(annotated_yolo_path, max_size=(1920, 1080), quality=85)
        send_photo(user_id, compressed_yolo, caption=yolo_report)

        img_original = cv2.imread(image_path)
        for i, det in enumerate(dets):
            bbox = det.get('bbox')
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img_original, str(i+1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        annotated_mllm_path = save_user_image(user_id, cv2.imencode('.jpg', img_original)[1].tobytes(), f"agentic_mllm_{task}_{obj}")

        compressed_mllm = compress_image(annotated_mllm_path, max_size=(512, 512), quality=70)
        with open(compressed_mllm, 'rb') as f:
            img_bytes_mllm = f.read()
        img_b64_mllm = base64.b64encode(img_bytes_mllm).decode('utf-8')

        count_lines = [f"{cls}: {cnt}" for cls, cnt in counts.items()]
        count_str = ", ".join(count_lines)
        prompt = f"На фото обнаружены объекты: {count_str}. Опиши, что ты видишь, дай рекомендации. Используй русский язык."

        send_message(user_id, "⏳ Анализирую изображение с помощью мультимодальной модели...")
        mllm_answer = await query_mllm(prompt, image_base64=img_b64_mllm,
                                       system_message="Ты полезный ассистент. Отвечай только на русском языке, кратко и по делу.")
        if not mllm_answer:
            mllm_answer = "❌ Не удалось получить ответ от MLLM."

        compressed_for_telegram = compress_image(annotated_mllm_path, max_size=(1920, 1080), quality=85)
        send_photo(user_id, compressed_for_telegram, caption=f"🤖 **Агентный анализ:**\n{mllm_answer}", keyboard=get_after_agentic_keyboard())

        user = get_user_state(user_id)
        user['agentic_last_image'] = compressed_mllm
        user['agentic_history'] = [{"role": "assistant", "content": mllm_answer}]
        user['step'] = None

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "task": f"agentic_{task}",
            "object": obj,
            "metadata": {},
            "weather": weather,
            "detections": dets,
            "counts": counts,
            "image_path": original_path,
            "annotated_path": annotated_mllm_path,
            "image_width": img_width,
            "image_height": img_height,
            "settings": settings,
            "agentic_answer": mllm_answer
        }
        save_analysis(user_id, analysis)

    except Exception as e:
        logger.exception("Ошибка при агентном анализе")
        send_message(user_id, f"❌ Ошибка: {type(e).__name__}")
    finally:
        for f in [compressed_yolo, compressed_mllm, annotated_yolo_path, annotated_mllm_path]:
            if f and os.path.exists(f):
                os.remove(f)

async def process_test_image_vk(user_id, image_path, model_name, original_path):
    settings = get_user_state(user_id)["settings"]
    model_rel_path = f"{user_id}/{model_name}"
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(image_path, 'rb') as f:
                resp = await client.post(
                    "http://127.0.0.1:9000/api/test_model",
                    files={'file': f},
                    data={
                        'model_path': model_rel_path,
                        'conf': settings['conf'],
                        'iou': settings['iou'],
                        'max_det': settings['max_det'],
                        'device': settings['device']
                    }
                )
        if resp.status_code != 200:
            send_message(user_id, f"❌ Ошибка сервера: {resp.status_code}")
            return
        data = resp.json()
        if "error" in data:
            send_message(user_id, f"❌ {data['error']}")
            return
        dets = data.get("detections", [])
        counts = data.get("counts", {})
        img_b64 = data["image_base64"]
        img_bytes = base64.b64decode(img_b64)
        annotated_path = save_user_image(user_id, img_bytes, f"test_{model_name.replace('.', '_')}")
        img_width = data.get("image_width", 640)
        img_height = data.get("image_height", 640)

        if not dets:
            report = "✅ Объектов не обнаружено."
        else:
            by_cls = defaultdict(list)
            for d in dets:
                by_cls[d['class']].append(d['confidence'])
            lines = ["🔎 **Обнаружено:**"]
            for cls, confs in by_cls.items():
                lines.append(f"- {cls}: {len(confs)} (ср. {sum(confs)/len(confs):.2f})")
            all_conf = [d['confidence'] for d in dets]
            lines.append(f"\n📊 Ср. уверенность: {sum(all_conf)/len(all_conf):.2f}")
            report = "\n".join(lines)
        report += f"\n⚙️ conf={settings['conf']}, iou={settings['iou']}, max_det={settings['max_det']}, device={settings['device']}"

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "task": "test_model",
            "object": model_name,
            "metadata": {},
            "weather": None,
            "detections": dets,
            "counts": counts,
            "image_path": original_path,
            "annotated_path": annotated_path,
            "image_width": img_width,
            "image_height": img_height,
            "settings": settings
        }
        save_analysis(user_id, analysis)

        user = get_user_state(user_id)
        user['last_detections'] = dets
        user['last_counts'] = counts
        user['last_metadata'] = {}
        user['last_weather'] = None
        user['last_original_path'] = original_path
        user['last_task'] = "test_model"
        user['last_object'] = model_name
        user['last_image_width'] = img_width
        user['last_image_height'] = img_height
        user['analysis_context'] = analysis
        user['step'] = None

        compressed = compress_image(annotated_path, max_size=(1920, 1080), quality=85)
        send_photo(user_id, compressed, caption=report, keyboard=get_after_photo_keyboard())

    except Exception as e:
        logger.exception("Ошибка теста модели")
        send_message(user_id, f"❌ Ошибка: {type(e).__name__}")

async def ask_about_result_async(user_id, text, user):
    analysis_ctx = user.get("analysis_context")
    if not analysis_ctx:
        send_message(user_id, "❌ Контекст анализа утерян.", get_main_keyboard())
        user["step"] = None
        return

    lines = ["Ты ассистент, отвечающий на вопросы по результатам анализа. Отвечай только на русском языке, кратко и по делу."]
    lines.append("Контекст анализа:")
    dets = analysis_ctx.get("detections", [])
    if dets:
        for d in dets:
            line = f"- {d['class']}"
            if 'confidence' in d:
                line += f" (уверенность {d['confidence']:.2f})"
            if 'count' in d:
                line += f": {d['count']}"
            lines.append(line)
    else:
        counts = analysis_ctx.get("counts", {})
        for cls, cnt in counts.items():
            lines.append(f"- {cls}: {cnt}")

    meta = analysis_ctx.get("metadata", {})
    if meta:
        if meta.get('sort'):
            lines.append(f"Сорт: {meta['sort']}")
        if meta.get('phase'):
            lines.append(f"Фаза: {meta['phase']}")
        if meta.get('comment'):
            lines.append(f"Комментарий: {meta['comment']}")
    weather = analysis_ctx.get("weather")
    if weather:
        lines.append(f"Погода: {weather['temp']}°C, {weather['description']}")

    sources_used = set()
    if user.get("ask_rag") and RAG_AVAILABLE:
        manager = RAGManager(user_id)
        docs = manager.search(text, top_k=2)
        if docs:
            lines.append("\nРелевантные фрагменты из базы знаний:")
            for i, doc in enumerate(docs, 1):
                lines.append(f"[{i}] (из документа {doc['source']}): {doc['text'][:200]}...")
                sources_used.add(doc['source'])

    system_msg = "\n".join(lines)

    history = user.get("ask_history", [])
    if len(history) > 20:
        history = history[-20:]

    messages = [{"role": "system", "content": system_msg}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }

    send_message(user_id, "⏳ Думаю...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(LLM_API_URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                history.append({"role": "user", "content": text})
                history.append({"role": "assistant", "content": answer})
                user["ask_history"] = history

                if sources_used:
                    answer += "\n\n📚 **Источники:** " + ", ".join(f"`{s}`" for s in sources_used)

                send_message(user_id, answer, get_back_keyboard())
            else:
                send_message(user_id, f"❌ Ошибка LLM: {resp.status_code}", get_back_keyboard())
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM: {e}")
        send_message(user_id, "❌ Ошибка при обращении к LLM.", get_back_keyboard())

async def agentic_dialog_async(user_id, text, user):
    last_image = user.get('agentic_last_image')
    if not last_image or not os.path.exists(last_image):
        send_message(user_id, "❌ Изображение не найдено.")
        user["step"] = None
        return
    with open(last_image, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    history = user.get('agentic_history', [])
    send_message(user_id, "⏳ Думаю...")
    answer = await query_mllm(text, image_base64=img_b64, history=history,
                             system_message="Ты полезный ассистент. Отвечай только на русском языке, кратко и по делу.")
    if answer:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
        user['agentic_history'] = history
        send_message(user_id, answer, get_back_keyboard())
    else:
        send_message(user_id, "❌ Ошибка при обращении к MLLM.", get_back_keyboard())

# ================== ОСНОВНОЙ ЦИКЛ ==================
if __name__ == "__main__":
    try:
        r = requests.get("http://127.0.0.1:9000/health", timeout=2)
        if r.status_code == 200:
            logger.info("✅ Сервер анализа доступен")
        else:
            logger.warning("⚠️ Сервер анализа ответил странным кодом")
    except:
        logger.warning("⚠️ Сервер анализа НЕ ДОСТУПЕН. Запустите его: uvicorn server:app --port 9000")

    print("🚀 VK Bot запущен. Ожидание сообщений...")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            text = event.text.strip()
            user = get_user_state(user_id)
            step = user.get("step")

            logger.info(f"Сообщение от {user_id}: {text}")

            attachments = []
            if hasattr(event, 'message_id') and event.message_id:
                try:
                    messages = vk.messages.getById(message_ids=event.message_id)
                    if messages and messages.get('count', 0) > 0:
                        msg = messages['items'][0]
                        attachments = msg.get('attachments', [])
                except Exception as e:
                    logger.error(f"Ошибка получения сообщения через API: {e}")

            if attachments:
                photo_bytes = get_photo_bytes_from_attachments(attachments)
                if photo_bytes:
                    tmp_path = f"/tmp/vk_{user_id}_{datetime.now().timestamp()}.jpg"
                    with open(tmp_path, "wb") as f:
                        f.write(photo_bytes)

                    if step == PHOTO_WAITING_PHOTO:
                        task = user.get("photo_task")
                        obj = user.get("photo_object")
                        if not task or not obj:
                            send_message(user_id, "❌ Сначала выберите задачу и объект.")
                            os.remove(tmp_path)
                            continue
                        meta = user.get("metadata", {})
                        sett = user.get("settings", DEFAULT_SETTINGS.copy())
                        orig_path = save_user_image(user_id, photo_bytes, "vk_original")
                        asyncio.run(process_image_vk(user_id, tmp_path, task, obj, meta, orig_path, sett))
                        os.remove(tmp_path)
                        continue

                    elif step == AGENTIC_WAITING_PHOTO:
                        task = user.get("agentic_task")
                        obj = user.get("agentic_object")
                        if not task or not obj:
                            send_message(user_id, "❌ Сначала выберите задачу и объект.")
                            os.remove(tmp_path)
                            continue
                        orig_path = save_user_image(user_id, photo_bytes, "vk_agentic_original")
                        asyncio.run(process_agentic_image_vk(user_id, tmp_path, task, obj, orig_path))
                        os.remove(tmp_path)
                        continue

                    elif step == DATASET_WAITING_PHOTOS:
                        dm = DatasetManager(user_id)
                        task = dm.metadata.get("task")
                        obj = dm.metadata.get("object")
                        if not task or not obj:
                            send_message(user_id, "❌ Сначала выберите задачу для датасета.")
                            os.remove(tmp_path)
                            continue
                        settings = user.get("settings", DEFAULT_SETTINGS.copy())
                        metadata = user.get("metadata", {})
                        weather = None
                        if metadata.get("latitude") and metadata.get("longitude"):
                            weather = asyncio.run(get_weather(metadata["latitude"], metadata["longitude"]))

                        annotated_tmp = None
                        compressed_annotated = None
                        try:
                            async def analyze():
                                async with httpx.AsyncClient(timeout=300) as client:
                                    with open(tmp_path, 'rb') as f:
                                        resp = await client.post(
                                            "http://127.0.0.1:9000/api/analyze_plant",
                                            files={'file': f},
                                            data={
                                                'task': task,
                                                'object': obj,
                                                'conf': settings['conf'],
                                                'iou': settings['iou'],
                                                'max_det': settings['max_det'],
                                                'device': settings['device']
                                            }
                                        )
                                    return resp
                            resp = asyncio.run(analyze())
                            if resp.status_code != 200:
                                send_message(user_id, f"❌ Ошибка сервера: {resp.status_code}")
                                os.remove(tmp_path)
                                continue
                            data = resp.json()
                            if "error" in data:
                                send_message(user_id, f"❌ {data['error']}")
                                os.remove(tmp_path)
                                continue
                            dets = data.get("detections", [])
                            img_b64 = data["image_base64"]
                            img_bytes_annotated = base64.b64decode(img_b64)
                            img_width = data.get("image_width", 640)
                            img_height = data.get("image_height", 640)

                            annotated_tmp = f"/tmp/{user_id}_{datetime.now().timestamp()}_annotated.jpg"
                            with open(annotated_tmp, 'wb') as f:
                                f.write(img_bytes_annotated)
                            compressed_annotated = compress_image(annotated_tmp, max_size=(1280, 1280), quality=80)

                            send_message(user_id, "🔍 Результат разметки для датасета:")
                            send_photo(user_id, compressed_annotated)

                            json_str = json.dumps(dets, ensure_ascii=False, indent=2)
                            json_bytes = json_str.encode('utf-8')
                            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
                                tmp_json.write(json_bytes)
                                tmp_json.flush()
                                send_document(user_id, tmp_json.name, "detections.json")
                                os.unlink(tmp_json.name)

                            with open(tmp_path, 'rb') as f:
                                img_bytes_original = f.read()
                            user["current_detections"] = dets
                            user["current_image_bytes"] = img_bytes_original
                            user["current_image_width"] = img_width
                            user["current_image_height"] = img_height
                            user["weather"] = weather
                            user["step"] = DATASET_CONFIRM_ANNOTATION
                            send_message(user_id, "Сохранить эту разметку в датасет?", get_dataset_confirm_keyboard())

                        except Exception as e:
                            logger.exception("Ошибка датасета")
                            send_message(user_id, f"❌ Ошибка: {type(e).__name__}")
                        finally:
                            if annotated_tmp and os.path.exists(annotated_tmp):
                                os.remove(annotated_tmp)
                            if compressed_annotated and os.path.exists(compressed_annotated):
                                os.remove(compressed_annotated)
                        os.remove(tmp_path)
                        continue

                    elif step == DATASET_WAITING_PHOTOS_NO_ANNOT:
                        dm = DatasetManager(user_id)
                        task = dm.metadata.get("task")
                        obj = dm.metadata.get("object")
                        if not task or not obj:
                            send_message(user_id, "❌ Сначала выберите задачу для датасета.")
                            os.remove(tmp_path)
                            continue
                        metadata = user.get("metadata", {})
                        weather = None
                        if metadata.get("latitude") and metadata.get("longitude"):
                            weather = asyncio.run(get_weather(metadata["latitude"], metadata["longitude"]))
                        img = cv2.imread(tmp_path)
                        if img is None:
                            send_message(user_id, "❌ Не удалось прочитать изображение.")
                            os.remove(tmp_path)
                            continue
                        img_height, img_width = img.shape[:2]
                        with open(tmp_path, 'rb') as f:
                            img_bytes = f.read()
                        dm.add_image(img_bytes, [], img_width, img_height, metadata=metadata, weather=weather)
                        send_message(user_id, "✅ Изображение сохранено в датасет без разметки.")
                        send_message(user_id, "📸 Можете отправлять следующее фото.", get_back_keyboard())
                        os.remove(tmp_path)
                        continue

                    elif step == WAITING_TEST_PHOTO:
                        model_name = user.get("selected_model")
                        if not model_name:
                            send_message(user_id, "❌ Модель не выбрана.")
                            os.remove(tmp_path)
                            continue
                        orig_path = save_user_image(user_id, photo_bytes, "test_original")
                        asyncio.run(process_test_image_vk(user_id, tmp_path, model_name, orig_path))
                        os.remove(tmp_path)
                        continue

                    else:
                        send_message(user_id, "❌ Не выбрано действие. Нажмите /start")
                        os.remove(tmp_path)
                        continue

                for att in attachments:
                    if att.get('type') == 'doc':
                        doc = att.get('doc')
                        if not doc:
                            continue
                        filename = doc.get('title', '')
                        ext = os.path.splitext(filename)[1].lower()
                        doc_url = doc.get('url')
                        if not doc_url:
                            continue
                        doc_bytes = vk_session.http.get(doc_url).content
                        tmp_path = f"/tmp/vk_{user_id}_{datetime.now().timestamp()}_{filename}"
                        with open(tmp_path, "wb") as f:
                            f.write(doc_bytes)

                        if step == WAITING_MODEL_FILE:
                            if ext != '.pt':
                                send_message(user_id, "❌ Пожалуйста, отправьте файл с расширением .pt")
                                os.remove(tmp_path)
                                break
                            send_message(user_id, "⏳ Загружаю модель на сервер...")
                            try:
                                with open(tmp_path, 'rb') as f:
                                    resp = requests.post(
                                        "http://127.0.0.1:9000/api/upload_model",
                                        data={"user_id": str(user_id)},
                                        files={"file": (filename, f, "application/octet-stream")}
                                    )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    send_message(user_id, f"✅ Модель загружена: {data['filename']}")
                                    user["step"] = WAITING_MODEL_SELECTION
                                else:
                                    send_message(user_id, f"❌ Ошибка загрузки: {resp.text}")
                            except Exception as e:
                                logger.error(f"Ошибка загрузки модели: {e}")
                                send_message(user_id, "❌ Не удалось загрузить модель.")
                            os.remove(tmp_path)
                            break

                        if step == RAG_WAITING_DOCUMENT:
                            if ext not in ['.pdf', '.txt', '.docx']:
                                send_message(user_id, "❌ Поддерживаются только PDF, TXT, DOCX")
                                os.remove(tmp_path)
                                break
                            send_message(user_id, "⏳ Обрабатываю документ...")
                            manager = RAGManager(user_id)
                            success = manager.add_document(tmp_path, filename)
                            if success:
                                send_message(user_id, "✅ Документ добавлен в базу знаний.")
                            else:
                                send_message(user_id, "❌ Не удалось обработать документ.")
                            os.remove(tmp_path)
                            user["step"] = RAG_MENU
                            send_message(user_id, "📚 Меню RAG", get_rag_menu_keyboard())
                            break

                        if ext in {'.mp4', '.mov', '.avi', '.mkv'}:
                            if step != VIDEO_READY:
                                send_message(user_id, "❌ Сначала выберите задачу и объект для видео.")
                                os.remove(tmp_path)
                                break
                            # Здесь можно добавить реальную обработку видео, но в данной версии пока нет
                            send_message(user_id, "⚠️ Анализ видео пока не реализован. Используйте анализ изображений.")
                            os.remove(tmp_path)
                            break

                        image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
                        if ext in image_exts:
                            pass
                        break

            if user.get('analysis_context') and step is None:
                if text == "🤖 Описать результат":
                    dets = user.get('last_detections', [])
                    if not dets:
                        send_message(user_id, "❌ Нет данных для описания.")
                        continue
                    lines = ["Результаты анализа:"]
                    if dets:
                        by_cls = defaultdict(list)
                        for d in dets:
                            by_cls[d['class']].append(d['confidence'])
                        for cls, confs in by_cls.items():
                            avg_conf = sum(confs) / len(confs)
                            lines.append(f"- {cls}: {len(confs)} (ср. уверенность {avg_conf:.2f})")
                    meta = user.get('last_metadata', {})
                    if meta.get('sort'):
                        lines.append(f"Сорт: {meta['sort']}")
                    if meta.get('phase'):
                        lines.append(f"Фаза: {meta['phase']}")
                    if meta.get('comment'):
                        lines.append(f"Комментарий: {meta['comment']}")
                    weather = user.get('last_weather')
                    if weather:
                        lines.append(f"Погода: {weather['temp']}°C, {weather['description']}")
                    prompt = "\n".join(lines) + "\n\nОпиши кратко, что обнаружено, и дай рекомендации."
                    send_message(user_id, "⏳ Генерирую описание...")
                    answer = asyncio.run(query_llm(prompt))
                    if answer:
                        send_message(user_id, f"🤖 **Описание результата:**\n{answer}")
                    else:
                        send_message(user_id, "❌ Не удалось получить ответ от LLM.")
                    continue

                elif text == "💬 Задать вопрос":
                    if not user.get('analysis_context'):
                        send_message(user_id, "❌ Нет данных для вопросов.")
                        continue
                    user["step"] = ASK_ABOUT_RESULT
                    user["ask_history"] = []
                    user.pop("ask_rag", None)
                    send_message(user_id, "💬 Задавайте вопросы по результатам анализа. Для выхода напишите /exit или нажмите кнопку ниже.", get_back_keyboard())
                    continue

                elif text == "🔍 Задать вопрос с RAG":
                    if not RAG_AVAILABLE:
                        send_message(user_id, "❌ RAG не доступен (не установлены зависимости).")
                        continue
                    if not user.get('analysis_context'):
                        send_message(user_id, "❌ Нет данных для вопросов.")
                        continue
                    user["step"] = ASK_ABOUT_RESULT
                    user["ask_rag"] = True
                    user["ask_history"] = []
                    send_message(user_id, "💬 Задавайте вопросы с использованием базы знаний. Для выхода напишите /exit.", get_back_keyboard())
                    continue

                elif text == "📥 Экспорт":
                    send_message(user_id, "Функция экспорта доступна в разделе Инструменты ➡️ Экспорт CSV.")
                    continue

            if step is None and text == "💬 Задать вопрос по изображению":
                if not user.get('agentic_last_image'):
                    send_message(user_id, "❌ Нет данных для вопросов. Сначала выполните агентный анализ.")
                    continue
                user["step"] = AGENTIC_DIALOG
                send_message(user_id, "💬 Задавайте вопросы по изображению. Для выхода напишите /exit.", get_back_keyboard())
                continue

            if step == AGENTIC_DIALOG:
                if text == "/exit" or text == "⬅️ Назад":
                    user["step"] = None
                    send_message(user_id, "Выход из диалога.", get_main_keyboard())
                    continue
                asyncio.run(agentic_dialog_async(user_id, text, user))
                continue

            if text == "/start" or text == "Начать":
                user.clear()
                user["settings"] = DEFAULT_SETTINGS.copy()
                user["step"] = None
                send_message(user_id, "🍏 AI-ассистент садовода (VK версия)\nВыберите действие:", get_main_keyboard())
                continue

            if text == "⬅️ Назад":
                user["step"] = None
                send_message(user_id, "Главное меню", get_main_keyboard())
                continue

            if text == "📷 Анализ":
                user["step"] = ANALYSIS_MENU
                send_message(user_id, "Выберите тип анализа:", get_analysis_keyboard())
                continue

            if text == "🤖 AI":
                user["step"] = AI_MENU
                send_message(user_id, "Выберите AI-функцию:", get_ai_keyboard())
                continue

            if text == "📊 Данные":
                user["step"] = DATA_MENU
                send_message(user_id, "Управление данными:", get_data_keyboard())
                continue

            if text == "🛠️ Инструменты":
                user["step"] = TOOLS_MENU
                send_message(user_id, "Инструменты:", get_tools_keyboard())
                continue

            if text == "⚙️ Настройки":
                send_message(user_id, "⚙️ Настройки", get_settings_keyboard(user["settings"]))
                continue

            if text == "📞 Контакт":
                send_message(user_id, f"📞 @{ADMIN_USERNAME}")
                continue

            if user.get("step") == ANALYSIS_MENU:
                if text == "🖼️ Анализ изображений":
                    user["step"] = PHOTO_SELECT_TASK
                    task_names = list(TASKS.keys())
                    keyboard = VkKeyboard(inline=True)
                    for i in range(0, len(task_names), 2):
                        keyboard.add_button(task_names[i], color=VkKeyboardColor.PRIMARY)
                        if i+1 < len(task_names):
                            keyboard.add_button(task_names[i+1], color=VkKeyboardColor.PRIMARY)
                        keyboard.add_line()
                    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                    send_message(user_id, "Выберите задачу для анализа изображения:", keyboard.get_keyboard())
                    continue
                elif text == "🎥 Анализ видео":
                    user["step"] = VIDEO_SELECT_TASK
                    task_names = list(TASKS.keys())
                    keyboard = VkKeyboard(inline=True)
                    for i in range(0, len(task_names), 2):
                        keyboard.add_button(task_names[i], color=VkKeyboardColor.PRIMARY)
                        if i+1 < len(task_names):
                            keyboard.add_button(task_names[i+1], color=VkKeyboardColor.PRIMARY)
                        keyboard.add_line()
                    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                    send_message(user_id, "Выберите задачу для анализа видео:", keyboard.get_keyboard())
                    continue
                elif text == "⬅️ Назад":
                    user["step"] = None
                    send_message(user_id, "Главное меню", get_main_keyboard())
                    continue
                else:
                    send_message(user_id, "Неизвестная команда.")
                    continue

            if user.get("step") == AI_MENU:
                if text == "💬 Чат с LLM":
                    user["step"] = LLM_CHAT_MODE
                    send_message(user_id, "💬 Вы в режиме чата с LLM. Отправляйте текстовые сообщения.\nДля выхода напишите /exit или нажмите кнопку ниже.", get_back_keyboard())
                    continue
                elif text == "🤖 Agentic AI":
                    user["step"] = AGENTIC_SELECT_TASK
                    task_names = list(TASKS.keys())
                    keyboard = VkKeyboard(inline=True)
                    for i in range(0, len(task_names), 2):
                        keyboard.add_button(task_names[i], color=VkKeyboardColor.PRIMARY)
                        if i+1 < len(task_names):
                            keyboard.add_button(task_names[i+1], color=VkKeyboardColor.PRIMARY)
                        keyboard.add_line()
                    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                    send_message(user_id, "Выберите задачу для агентного анализа:", keyboard.get_keyboard())
                    continue
                elif text == "⬅️ Назад":
                    user["step"] = None
                    send_message(user_id, "Главное меню", get_main_keyboard())
                    continue
                else:
                    send_message(user_id, "Неизвестная команда.")
                    continue

            if user.get("step") == DATA_MENU:
                if text == "📁 Сбор датасета":
                    user["step"] = DATASET_MENU
                    send_message(user_id, "📊 Меню сбора датасета", get_dataset_menu_keyboard())
                    continue
                elif text == "📚 База знаний":
                    user["step"] = RAG_MENU
                    send_message(user_id, "📚 Управление базой знаний (RAG)", get_rag_menu_keyboard())
                    continue
                elif text == "⬅️ Назад":
                    user["step"] = None
                    send_message(user_id, "Главное меню", get_main_keyboard())
                    continue
                else:
                    send_message(user_id, "Неизвестная команда.")
                    continue

            if user.get("step") == TOOLS_MENU:
                if text == "🧪 Тест модели":
                    user["test_mode"] = True
                    user.pop("selected_model", None)
                    user["step"] = WAITING_MODEL_FILE
                    send_message(user_id, "📤 Отправьте файл модели (.pt) для загрузки.", get_back_keyboard())
                    continue
                elif text == "📊 Статистика":
                    p = asyncio.run(generate_stats(user_id))
                    if not p:
                        send_message(user_id, "📭 Нет данных для статистики.")
                    else:
                        send_photo(user_id, p, caption="📊 Статистика")
                        os.remove(p)
                    send_message(user_id, "⬅️ Меню:", get_main_keyboard())
                    continue
                elif text == "📋 История":
                    analyses = get_user_analyses(user_id, 10)
                    if not analyses:
                        send_message(user_id, "📭 Нет истории.")
                    else:
                        msg = "📋 Последние 10 анализов:\n"
                        for a in analyses:
                            dt = a.get('timestamp', '')[:16]
                            msg += f"• {dt} | {a.get('task')}/{a.get('object')} | объектов {len(a.get('detections', []))}\n"
                        send_message(user_id, msg)
                    send_message(user_id, "⬅️ Меню:", get_main_keyboard())
                    continue
                elif text == "📥 Экспорт CSV":
                    csv_path = export_to_csv(user_id)
                    if not csv_path:
                        send_message(user_id, "📭 Нет данных для экспорта.")
                    else:
                        send_document(user_id, csv_path, f"analyses_{user_id}.csv")
                        os.remove(csv_path)
                    send_message(user_id, "⬅️ Меню:", get_main_keyboard())
                    continue
                elif text == "⬅️ Назад":
                    user["step"] = None
                    send_message(user_id, "Главное меню", get_main_keyboard())
                    continue
                else:
                    send_message(user_id, "Неизвестная команда.")
                    continue

            if step == PHOTO_SELECT_TASK:
                for name, tid in TASKS.items():
                    if text == name:
                        user["photo_task"] = tid
                        user["step"] = PHOTO_SELECT_OBJECT
                        allowed = get_allowed_objects(tid)
                        obj_names = list(allowed.keys())
                        user["object_list"] = obj_names
                        user["object_page"] = 0
                        keyboard = VkKeyboard(inline=True)
                        start = 0
                        page_size = 6
                        page_obj = obj_names[start:start+page_size]
                        for i in range(0, len(page_obj), 2):
                            keyboard.add_button(page_obj[i], color=VkKeyboardColor.PRIMARY)
                            if i+1 < len(page_obj):
                                keyboard.add_button(page_obj[i+1], color=VkKeyboardColor.PRIMARY)
                            keyboard.add_line()
                        if start + page_size < len(obj_names):
                            keyboard.add_button("➡️ Далее", color=VkKeyboardColor.SECONDARY)
                            keyboard.add_line()
                        keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                        send_message(user_id, "Выберите объект для анализа изображения (страница 1):", keyboard.get_keyboard())
                        break
                else:
                    send_message(user_id, "Неизвестная задача.")
                continue

            if step == PHOTO_SELECT_OBJECT:
                if text == "➡️ Далее":
                    page = user.get("object_page", 0) + 1
                    obj_names = user.get("object_list", [])
                    if obj_names:
                        start = page * 6
                        end = start + 6
                        page_obj = obj_names[start:end]
                        keyboard = VkKeyboard(inline=True)
                        for i in range(0, len(page_obj), 2):
                            keyboard.add_button(page_obj[i], color=VkKeyboardColor.PRIMARY)
                            if i+1 < len(page_obj):
                                keyboard.add_button(page_obj[i+1], color=VkKeyboardColor.PRIMARY)
                            keyboard.add_line()
                        if start > 0:
                            keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                            keyboard.add_line()
                        if end < len(obj_names):
                            keyboard.add_button("➡️ Далее", color=VkKeyboardColor.SECONDARY)
                            keyboard.add_line()
                        keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                        send_message(user_id, f"Выберите объект (страница {page+1}):", keyboard.get_keyboard())
                        user["object_page"] = page
                    continue
                if text == "⬅️ Назад" and text != "⬅️ Назад":
                    # это не сработает, потому что выше есть обработка "⬅️ Назад", а тут нужна логика возврата страницы
                    # Для корректности, лучше явно различать кнопки пагинации и глобальное возвращение
                    pass
                if text == "⬅️ Назад":
                    user["step"] = PHOTO_SELECT_TASK
                    task_names = list(TASKS.keys())
                    keyboard = VkKeyboard(inline=True)
                    for i in range(0, len(task_names), 2):
                        keyboard.add_button(task_names[i], color=VkKeyboardColor.PRIMARY)
                        if i+1 < len(task_names):
                            keyboard.add_button(task_names[i+1], color=VkKeyboardColor.PRIMARY)
                        keyboard.add_line()
                    keyboard.add_button("⬅️ Назад", color=VkKeyboardColor.SECONDARY)
                    send_message(user_id, "Выберите задачу для анализа изображения:", keyboard.get_keyboard())
                    continue
                task = user.get("photo_task")
                if task:
                    allowed = get_allowed_objects(task)
                    for obj_name, obj_code in allowed.items():
                        if text == obj_name:
                            user["photo_object"] = obj_code
                            user["step"] = ASK_METADATA
                            send_message(user_id, "Хотите добавить метаданные?", get_metadata_ask_keyboard())
                            break
                else:
                    send_message(user_id, "Ошибка: задача не выбрана.")
                continue

            send_message(user_id, "Нажмите /start", get_main_keyboard())
