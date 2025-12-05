import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from datetime import datetime
import sqlite3
import json
from werkzeug.utils import secure_filename

# Инициализация Flask приложения
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Создание папок, если они не существуют
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Загрузка модели YOLOv8
print("Загрузка модели YOLOv8...")
try:
    model = YOLO('yolov8n.pt')  # Используем предобученную модель
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Таблица для истории запросов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            cups INTEGER DEFAULT 0,
            glasses INTEGER DEFAULT 0,
            mugs INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            processing_time REAL,
            result_image TEXT
        )
    ''')
    
    # Таблица для статистики
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_objects INTEGER DEFAULT 0,
            total_images INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("База данных инициализирована")

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Страница истории
@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT 50')
    records = cursor.fetchall()
    conn.close()
    
    # Подсчет общей статистики
    total_images = len(records)
    total_objects = sum(record[6] for record in records)  # total поле
    
    return render_template('history.html', 
                         records=records, 
                         total_images=total_images, 
                         total_objects=total_objects)

# Обработка изображения
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Сохранение оригинального файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # Загрузка и обработка изображения
    start_time = datetime.now()
    
    try:
        # Чтение изображения
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Изменение размера для более быстрой обработки (опционально)
        height, width = img.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Детекция объектов
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        results = model(img)
        
        # Анализ результатов
        cups = 0
        glasses = 0
        mugs = 0
        detected_objects = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Классы, которые нас интересуют
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Фильтрация по уверенности
                if confidence > 0.5:
                    if class_name in ['cup', 'wine glass', 'bottle']:
                        cups += 1
                        detected_objects.append({
                            'type': 'cup',
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
                    elif class_name in ['vase', 'bowl']:
                        glasses += 1
                        detected_objects.append({
                            'type': 'glass',
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
                    else:
                        mugs += 1
                        detected_objects.append({
                            'type': 'mug',
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
        
        # Визуализация результатов
        result_img = result.plot() if results else img
        
        # Сохранение результата
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        # Расчет времени обработки
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Сохранение в базу данных
        total_objects = cups + glasses + mugs
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history 
            (timestamp, filename, cups, glasses, mugs, total, processing_time, result_image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            filename,
            cups,
            glasses,
            mugs,
            total_objects,
            processing_time,
            result_filename
        ))
        conn.commit()
        conn.close()
        
        # Формирование ответа
        response = {
            'success': True,
            'filename': filename,
            'result_image': f'/static/results/{result_filename}',
            'original_image': f'/static/uploads/{filename}',
            'statistics': {
                'cups': cups,
                'glasses': glasses,
                'mugs': mugs,
                'total': total_objects
            },
            'processing_time': processing_time,
            'detected_objects': detected_objects
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

# Получение статистики
@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Общая статистика
    cursor.execute('SELECT COUNT(*) as total_images, SUM(total) as total_objects FROM history')
    stats = cursor.fetchone()
    
    # Статистика по дням
    cursor.execute('''
        SELECT DATE(timestamp) as date, 
               COUNT(*) as images_per_day,
               SUM(total) as objects_per_day
        FROM history 
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 7
    ''')
    daily_stats = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_images': stats[0] or 0,
        'total_objects': stats[1] or 0,
        'daily_stats': [
            {'date': row[0], 'images': row[1], 'objects': row[2]}
            for row in daily_stats
        ]
    })

# Экспорт истории в JSON
@app.route('/export/json')
def export_json():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()
    
    # Преобразование в список словарей
    data = []
    for record in records:
        data.append({
            'id': record[0],
            'timestamp': record[1],
            'filename': record[2],
            'cups': record[3],
            'glasses': record[4],
            'mugs': record[5],
            'total': record[6],
            'processing_time': record[7],
            'result_image': record[8]
        })
    
    return jsonify(data)

# Удаление записи
@app.route('/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

# Запуск приложения
if __name__ == '__main__':
    # Инициализация базы данных
    init_db()
    
    # Запуск Flask приложения
    print("\n" + "="*50)
    print("СИСТЕМА ПОДСЧЕТА ПОСУДЫ В КОФЕЙНЕ")
    print("="*50)
    print("\nДоступные маршруты:")
    print("  • Главная страница: http://localhost:5000")
    print("  • История запросов: http://localhost:5000/history")
    print("  • API статистики: http://localhost:5000/api/stats")
    print("  • Экспорт JSON: http://localhost:5000/export/json")
    print("\nДля остановки сервера нажмите Ctrl+C")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)