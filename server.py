#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import numpy as np
import cv2
import logging
import requests
import tempfile
import math
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from ultralytics import YOLO
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Загрузка моделей ==================
TASK_MODEL_BASENAME = {
    "detection": "yolo.pt",
    "instance_seg": "yolo-seg.pt",
    "semantic_seg": "yolo-seg.pt",
    "classification": "yolo-cls.pt",
    "keypoint": "yolo-pose.pt",
    "multimodal": "yolo.pt"
}

OBJECTS = [
    "fruit", "flower", "ovary", "disease", "general",
    "apple_night", "apple_decline", "apple_buds",
    "apple_ovary", "apple_pollinator_tree", "apple_tree_trunk",
    "apple_tree_pose"
]

MODEL_PATHS = {}
for task, basename in TASK_MODEL_BASENAME.items():
    for obj in OBJECTS:
        key = f"{task}_{obj}"
        MODEL_PATHS[key] = os.path.join("models", basename)

MODEL_PATHS["detection_fruit"] = os.path.join("models", "apple_bb.pt")
MODEL_PATHS["detection_disease"] = os.path.join("models", "apple_leaves_bb.pt")
MODEL_PATHS["detection_apple_night"] = os.path.join("models", "apple_night_bb.pt")
MODEL_PATHS["detection_apple_decline"] = os.path.join("models", "apple_decline_bb.pt")
MODEL_PATHS["detection_apple_buds"] = os.path.join("models", "apple_buds_bb.pt")
MODEL_PATHS["detection_apple_ovary"] = os.path.join("models", "apple_ovary_bb.pt")
MODEL_PATHS["detection_apple_pollinator_tree"] = os.path.join("models", "apple_pollinator_tree_bb.pt")
MODEL_PATHS["detection_apple_tree_trunk"] = os.path.join("models", "apple_tree_trunk_bb.pt")
MODEL_PATHS["detection_apple_flower"] = os.path.join("models", "apple_flower_bb.pt")
MODEL_PATHS["detection_fruit_flies"] = os.path.join("models", "fruit_flies.pt")
MODEL_PATHS["instance_seg_apple_flower"] = os.path.join("models", "apple_flower_seg.pt")
MODEL_PATHS["instance_seg_apple_fruit"] = os.path.join("models", "apple_seg.pt")
MODEL_PATHS["instance_seg_apple_sort"] = os.path.join("models", "apple_sort_seg.pt")
MODEL_PATHS["instance_seg_microscope_scab_powdery_mildew"] = os.path.join("models", "microscope_scab_powdery mildew_seg.pt")
MODEL_PATHS["keypoint_apple_tree_pose"] = os.path.join("models", "apple_tree_pose.pt")

loaded_models = {}
USER_MODELS_DIR = "user_models"
USER_VIDEOS_ORIGINAL_DIR = "user_data/videos_original"
USER_VIDEOS_ANNOTATED_DIR = "user_data/videos_annotated"
os.makedirs(USER_MODELS_DIR, exist_ok=True)
os.makedirs(USER_VIDEOS_ORIGINAL_DIR, exist_ok=True)
os.makedirs(USER_VIDEOS_ANNOTATED_DIR, exist_ok=True)

TREE_SKELETON = [(0,1), (1,2), (2,3), (3,4), (4,5)]

def draw_skeleton(image, kpts_with_conf, skeleton, color=(0,255,0), thickness=2, conf_threshold=0.5):
    for a, b in skeleton:
        if a < kpts_with_conf.shape[0] and b < kpts_with_conf.shape[0]:
            if kpts_with_conf[a,2] >= conf_threshold and kpts_with_conf[b,2] >= conf_threshold:
                pt1 = (int(kpts_with_conf[a,0]), int(kpts_with_conf[a,1]))
                pt2 = (int(kpts_with_conf[b,0]), int(kpts_with_conf[b,1]))
                cv2.line(image, pt1, pt2, color, thickness)

def get_yolo_model(task: str, obj: str, device: str = "cpu"):
    base_key = f"{task}_{obj}"
    if base_key not in MODEL_PATHS:
        if task in TASK_MODEL_BASENAME:
            path = os.path.join("models", TASK_MODEL_BASENAME[task])
        else:
            raise ValueError(f"Неизвестная комбинация: {task}_{obj}")
    else:
        path = MODEL_PATHS[base_key]
    cache_key = f"{base_key}_{device}"
    if cache_key not in loaded_models:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        logger.info(f"Загрузка модели {cache_key} из {path}")
        loaded_models[cache_key] = YOLO(path, task=task)
        if device != "cpu":
            loaded_models[cache_key].to(device)
    return loaded_models[cache_key]

app = FastAPI(title="AI Horticulture Assistant Server")

@app.post("/api/analyze_plant")
async def analyze_plant(
    file: UploadFile = File(...),
    task: str = Form(...),
    object: str = Form(...),
    conf: float = Form(0.15),
    iou: float = Form(0.45),
    max_det: int = Form(300),
    device: str = Form("cpu")
):
    try:
        model = get_yolo_model(task, object, device)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Ошибка загрузки модели: {str(e)}"})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})

        results = model(img, conf=conf, iou=iou, max_det=max_det)[0]
        annotated = results.plot()

        if task == "keypoint" and results.keypoints is not None:
            kpts_data = results.keypoints.data.cpu().numpy()
            for obj_kpts in kpts_data:
                draw_skeleton(annotated, obj_kpts, TREE_SKELETON)

        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        detections = []
        counts = {}
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = {"class": cls_name, "confidence": conf_val, "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                if results.masks is not None and i < len(results.masks.data):
                    mask = results.masks.data[i].cpu().numpy()
                    det["area"] = int(np.sum(mask))
                if results.keypoints is not None and i < len(results.keypoints.data):
                    det["keypoints"] = results.keypoints.data[i].cpu().numpy().tolist()
                detections.append(det)
                counts[cls_name] = counts.get(cls_name, 0) + 1

        return {
            "image_base64": img_base64,
            "detections": detections,
            "counts": counts,
            "image_width": img.shape[1],
            "image_height": img.shape[0]
        }
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/upload_model")
async def upload_model(file: UploadFile = File(...), user_id: str = Form(...)):
    if not file.filename.endswith('.pt'):
        raise HTTPException(400, "Файл должен иметь расширение .pt")
    user_dir = os.path.join(USER_MODELS_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(user_dir, safe_filename)
    if os.path.exists(file_path):
        base, ext = os.path.splitext(safe_filename)
        cnt = 1
        while os.path.exists(os.path.join(user_dir, f"{base}_{cnt}{ext}")):
            cnt += 1
        safe_filename = f"{base}_{cnt}{ext}"
        file_path = os.path.join(user_dir, safe_filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    return {"model_path": safe_filename, "filename": safe_filename}

@app.post("/api/test_model")
async def test_model(
    file: UploadFile = File(...),
    model_path: str = Form(...),
    conf: float = Form(0.15),
    iou: float = Form(0.45),
    max_det: int = Form(300),
    device: str = Form("cpu")
):
    full_path = os.path.join(USER_MODELS_DIR, model_path)
    if not os.path.isfile(full_path):
        raise HTTPException(404, "Файл модели не найден")
    model = YOLO(full_path)
    if device != "cpu":
        model.to(device)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})
    results = model(img, conf=conf, iou=iou, max_det=max_det)[0]
    annotated = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    detections = []
    counts = {}
    if results.boxes is not None:
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            counts[cls_name] = counts.get(cls_name, 0) + 1
            detections.append({
                "class": cls_name,
                "confidence": float(box.conf[0]),
                "bbox": [int(v) for v in box.xyxy[0].tolist()]
            })
    return {
        "image_base64": img_base64,
        "detections": detections,
        "counts": counts,
        "image_width": img.shape[1],
        "image_height": img.shape[0]
    }

@app.get("/api/model_info")
async def model_info(user_id: str, model_name: str):
    preset_path = os.path.join("models", model_name)
    if os.path.isfile(preset_path):
        stat = os.stat(preset_path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        return {
            "type": "preset",
            "name": model_name,
            "size": size,
            "modified": mtime,
            "classes": "unknown",
            "map": "unknown",
            "date": "unknown"
        }
    user_dir = os.path.join(USER_MODELS_DIR, str(user_id))
    file_path = os.path.join(user_dir, model_name)
    if not os.path.isfile(file_path):
        raise HTTPException(404, "Модель не найдена")
    stat = os.stat(file_path)
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
    return {"type": "user", "name": model_name, "size": size, "modified": mtime}

@app.delete("/api/delete_model")
async def delete_model(user_id: str, model_name: str):
    user_dir = os.path.join(USER_MODELS_DIR, str(user_id))
    file_path = os.path.join(user_dir, model_name)
    if not os.path.isfile(file_path):
        raise HTTPException(404, "Модель не найдена")
    os.remove(file_path)
    return {"status": "ok"}

UPDATE_MODELS_URL = os.getenv("UPDATE_MODELS_URL", "https://example.com/models/{filename}")

@app.post("/api/admin/update_models")
async def update_models():
    updated = []
    failed = []
    for filename in os.listdir("models"):
        if filename.endswith(".pt"):
            url = UPDATE_MODELS_URL.format(filename=filename)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    local_path = os.path.join("models", filename)
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    updated.append(filename)
                    keys_to_delete = [k for k in loaded_models if k.startswith(filename.replace('.pt',''))]
                    for k in keys_to_delete:
                        del loaded_models[k]
                    logger.info(f"Модель обновлена: {filename}")
                else:
                    failed.append(filename)
            except Exception as e:
                logger.error(f"Ошибка обновления {filename}: {e}")
                failed.append(filename)
    return {"updated": updated, "failed": failed}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    missing = []
    for key, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        logger.warning("Некоторые файлы моделей отсутствуют. Для работы бота загрузите их в папку models/")
    else:
        logger.info("Все модели найдены.")
    port = 9000
    uvicorn.run(app, host="127.0.0.1", port=port, timeout_keep_alive=120, timeout_graceful_shutdown=120)
