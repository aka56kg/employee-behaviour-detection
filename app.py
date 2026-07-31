import os
import cv2
import torch
import ffmpeg
import subprocess
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Регистрируем MIME-типы для видео
mimetypes.init()
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/webm', '.webm')
mimetypes.add_type('video/ogg', '.ogg')
mimetypes.add_type('video/quicktime', '.mov')
mimetypes.add_type('video/x-msvideo', '.avi')
mimetypes.add_type('video/x-matroska', '.mkv')

# Загрузка модели
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
        model_load_error = f"Не удалось загрузить модель. Убедитесь, что файл 'best.pt' находится в папке с программой. Ошибка: {str(e2)}"
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
    
    # Рисуем рамку
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Подпись
    label = f"{class_name} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Фон для подписи
    cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return image

def process_image(image_path):
    """Обработка изображения с корректной трансформацией координат"""
    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]
    
    # Конвертируем в RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    detections = []
    result_img = img_rgb.copy()
    
    print(f"\n{'='*60}")
    print(f"Файл: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if model_type == 'v8' and results[0].boxes is not None:
        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            x1_model = int(xyxy[0])
            y1_model = int(xyxy[1])
            x2_model = int(xyxy[2])
            y2_model = int(xyxy[3])
            
            print(f"\nОбъект {idx+1}: {class_name} (conf={conf:.3f})")
            print(f"  Модель: x1={x1_model}, y1={y1_model}, x2={x2_model}, y2={y2_model}")
            
            # ТРАНСФОРМАЦИЯ:
            # x1, y1 — правильные (левая верхняя точка)
            # Ширина = y2_model - y1_model (а не x2_model - x1_model!)
            # Высота = x2_model - x1_model (а не y2_model - y1_model!)
            
            x1 = x1_model
            y1 = y1_model
            
            # Правильные ширина и высота (переставлены местами)
            correct_width = y2_model - y1_model   # 346 - 84 = 262
            correct_height = x2_model - x1_model  # 1027 - 413 = 614
            
            x2 = x1 + correct_width   # 413 + 262 = 675
            y2 = y1 + correct_height  # 84 + 614 = 698
            
            print(f"  correct_width = {correct_width}")
            print(f"  correct_height = {correct_height}")
            print(f"  Исправлено: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"  Размер: {x2-x1} x {y2-y1}")
            
            draw_detection_rgb(result_img, x1, y1, x2, y2, class_name, conf, class_id)
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
    
    # Сохраняем результат
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{os.path.basename(image_path)}')
    cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n{'='*60}\n")
    
    return result_path, detections
    

def process_video(video_path):
    """Обработка видео с сохранением звука"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Временный файл без звука
    temp_output = os.path.join(app.config['UPLOAD_FOLDER'], 
                               f'temp_{os.path.splitext(os.path.basename(video_path))[0]}.mp4')
    final_output = os.path.join(app.config['UPLOAD_FOLDER'], 
                                f'result_{os.path.splitext(os.path.basename(video_path))[0]}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
                
                # Коррекция координат (ваша формула)
                x1 = int(x1_orig)
                y1 = int(y1_orig)
                x2 = int(x1_orig + (y2_orig - y1_orig))
                y2 = int(y1_orig + (x2_orig - x1_orig))
                
                draw_detection_rgb(processed_frame_rgb, x1, y1, x2, y2, class_name, conf, class_id)
        
        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(processed_frame_bgr)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Обработано кадров: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Копируем звук из исходного видео в обработанное
    try:
        # Используем ffmpeg для копирования аудиодорожки
        audio_input = ffmpeg.input(video_path)
        video_input = ffmpeg.input(temp_output)
        
        ffmpeg.output(
            video_input.video, 
            audio_input.audio, 
            final_output,
            vcodec='copy',      # копируем видео без перекодирования
            acodec='aac',       # аудиокодек
            strict='experimental'
        ).run(overwrite_output=True, quiet=True)
        
        # Удаляем временный файл
        os.remove(temp_output)
        print(f"Звук успешно добавлен в {final_output}")
        
    except Exception as e:
        print(f"Не удалось добавить звук: {e}")
        # Если ffmpeg не сработал, используем временный файл как результат
        os.rename(temp_output, final_output)
    
    return final_output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_model', methods=['GET'])
def check_model():
    """Проверка наличия модели"""
    if model is None:
        return jsonify({'error': model_load_error or 'Модель не загружена'}), 500
    return jsonify({'status': 'ok', 'model_type': model_type})

@app.route('/upload', methods=['POST'])
def upload():
    if model is None:
        return jsonify({'error': model_load_error or 'Модель не загружена'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    allowed_image = ['jpg', 'jpeg', 'png', 'bmp']
    allowed_video = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'ogg']
    
    if ext not in allowed_image and ext not in allowed_video:
        return jsonify({'error': f'Неподдерживаемый формат: .{ext}'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        if ext in allowed_image:
            result_path, detections = process_image(filepath)
            return jsonify({
                'type': 'image',
                'result_url': f'/result/{os.path.basename(result_path)}',
                'detections': detections,
                'count': len(detections)
            })
        else:
            result_path = process_video(filepath)
            return jsonify({
                'type': 'video',
                'result_url': f'/result/{os.path.basename(result_path)}',
                'message': 'Видео обработано'
            })
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    mime_types = {
        'mp4': 'video/mp4', 'webm': 'video/webm', 'ogg': 'video/ogg',
        'mov': 'video/quicktime', 'avi': 'video/x-msvideo', 'mkv': 'video/x-matroska',
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'bmp': 'image/bmp'
    }
    
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    return send_file(filepath, mimetype=mime_type, as_attachment=False, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)