import os
import cv2
import torch
import ffmpeg
import numpy as np
import mimetypes
from pathlib import Path
import traceback

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename

app = FastAPI(title="YOLOv8 Detection Service")

# Настройка папок
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Монтирование статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Регистрируем MIME-типы для видео
mimetypes.init()
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/webm', '.webm')
mimetypes.add_type('video/ogg', '.ogg')
mimetypes.add_type('video/quicktime', '.mov')
mimetypes.add_type('video/x-msvideo', '.avi')
mimetypes.add_type('video/x-matroska', '.mkv')

# Загрузка модели PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

model = None
model_type = None
model_load_error = None

try:
    from ultralytics import YOLO
    model = YOLO('best_3.pt')
    model_type = 'v8'
    print("Загружена YOLOv8 модель")
except Exception as e:
    print(f"Ошибка загрузки YOLOv8: {e}")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model_type = 'v5'
        print("Загружена YOLOv5 модель")
    except Exception as e2:
        model_load_error = f"Не удалось загрузить модель. Убедитесь, что файл 'best_3.pt' или 'best.pt' находится в папке с программой. Ошибка: {str(e2)}"
        print(model_load_error)
        model = None
        model_type = None

# Цвета для рамок (в формате RGB)
CLASS_COLORS = {
    0: (255, 0, 0),      # Пустая комната - красный
    1: (0, 255, 0),      # Стоит/ходит - зеленый
    2: (0, 255, 255),    # Сидит - желтый
    3: (255, 0, 255),    # Разговаривает - пурпурный
    4: (0, 0, 255),      # Упал - синий
}
DEFAULT_COLOR = (0, 255, 0)


def draw_detection_rgb(image, x1, y1, x2, y2, class_name, confidence, class_id=None):
    """Рисует рамку и подпись на RGB изображении"""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    color = CLASS_COLORS.get(class_id, DEFAULT_COLOR) if class_id is not None else DEFAULT_COLOR
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    label = f"{class_name} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return image


def process_image(image_path: str):
    """Обработка изображения"""
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    detections = []
    result_img = img_rgb.copy()
    
    if model_type == 'v8' and results[0].boxes is not None:
        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            x1_model, y1_model, x2_model, y2_model = map(int, xyxy[:4])
            
            x1 = x1_model
            y1 = y1_model
            correct_width = y2_model - y1_model
            correct_height = x2_model - x1_model
            
            x2 = x1 + correct_width
            y2 = y1 + correct_height
            
            draw_detection_rgb(result_img, x1, y1, x2, y2, class_name, conf, class_id)
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
    
    filename = os.path.basename(image_path)
    result_path = UPLOAD_FOLDER / f"result_{filename}"
    cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return str(result_path), detections


def process_video(video_path: str):
    """Обработка видео с сохранением звука"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 60
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_output = str(UPLOAD_FOLDER / f"temp_{base_name}.mp4")
    final_output = str(UPLOAD_FOLDER / f"result_{base_name}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        processed_frame_rgb = frame_rgb.copy()
        
        if model_type == 'v8' and results[0].boxes is not None:
            for box in results[0].boxes:
                x1_orig, y1_orig, x2_orig, y2_orig = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                x1 = int(x1_orig)
                y1 = int(y1_orig)
                x2 = int(x1_orig + (y2_orig - y1_orig))
                y2 = int(y1_orig + (x2_orig - x1_orig))
                
                draw_detection_rgb(processed_frame_rgb, x1, y1, x2, y2, class_name, conf, class_id)
        
        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(processed_frame_bgr)
    
    cap.release()
    out.release()
    
    # Копируем звук и ПЕРЕКОДИРУЕМ видео в H.264 для поддержки браузерами
    try:
        audio_input = ffmpeg.input(video_path)
        video_input = ffmpeg.input(temp_output)
        
        ffmpeg.output(
            video_input.video, 
            audio_input.audio, 
            final_output,
            vcodec='libx264',      # Использование совместимого видеокодека H.264
            acodec='aac',          # Аудиокодек AAC
            pix_fmt='yuv420p',     # Обязательно для воспроизведения в браузерах!
            strict='experimental'
        ).run(overwrite_output=True, quiet=True)
        
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
    except Exception as e:
        print(f"Ошибка при обработке через ffmpeg (пробуем без аудио): {e}")
        # Если у исходного видео нет аудиодорожки, обработка выше может выдать ошибку.
        # В таком случае перекодируем только видео:
        try:
            ffmpeg.input(temp_output).output(
                final_output,
                vcodec='libx264',
                pix_fmt='yuv420p'
            ).run(overwrite_output=True, quiet=True)
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e2:
            print(f"Критическая ошибка перекодирования: {e2}")
            if os.path.exists(temp_output):
                os.rename(temp_output, final_output)
    
    return final_output


# Маршруты приложений

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница со статикой HTML"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/check_model")
async def check_model():
    """Проверка наличия модели"""
    if model is None:
        raise HTTPException(status_code=500, detail=model_load_error or 'Модель не загружена')
    return {"status": "ok", "model_type": model_type}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Загрузка и обработка файла"""
    if model is None:
        raise HTTPException(status_code=500, detail=model_load_error or 'Модель не загружена')
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    allowed_image = ['jpg', 'jpeg', 'png', 'bmp']
    allowed_video = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'ogg']
    
    if ext not in allowed_image and ext not in allowed_video:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: .{ext}")
    
    filepath = UPLOAD_FOLDER / filename
    
    # Записываем байты файла
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
    
    try:
        if ext in allowed_image:
            result_path, detections = process_image(str(filepath))
            return {
                'type': 'image',
                'result_url': f'/result/{os.path.basename(result_path)}',
                'detections': detections,
                'count': len(detections)
            }
        else:
            result_path = process_video(str(filepath))
            return {
                'type': 'video',
                'result_url': f'/result/{os.path.basename(result_path)}',
                'message': 'Видео обработано'
            }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/result/{filename}")
async def get_result(filename: str):
    """Отдача результатов загруженных файлов"""
    filepath = UPLOAD_FOLDER / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
        
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    mime_types = {
        'mp4': 'video/mp4', 'webm': 'video/webm', 'ogg': 'video/ogg',
        'mov': 'video/quicktime', 'avi': 'video/x-msvideo', 'mkv': 'video/x-matroska',
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'bmp': 'image/bmp'
    }
    
    mime_type = mime_types.get(ext, 'application/octet-stream')
    return FileResponse(path=filepath, media_type=mime_type, filename=filename)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)