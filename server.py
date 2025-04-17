import os
import cv2
import numpy as np
import sqlite3
import json
import re
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from functools import lru_cache
import logging
from pathlib import Path
import tensorflow as tf
from datetime import datetime
import shutil
from contextlib import contextmanager

# ==============================================
# НАСТРОЙКИ И КОНФИГУРАЦИЯ
# ==============================================

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alcohol_detector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Настройки базы данных
SQLITE_TIMEOUT = 15  # секунд
MAX_RETRIES = 5
RETRY_DELAY = 0.2  # секунд

# Разрешенные расширения для фото
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)

# Конфигурация путей
BASE_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = Path("C:/my_project2")
CLASSES_DIR = PROJECT_ROOT / "classes"
MODELS_DIR = PROJECT_ROOT / "Keras"
LEARNING_DIR = PROJECT_ROOT / "alcogole"
DATABASE_PATH = BASE_DIR / "bottle.db"
UPLOAD_FOLDER = BASE_DIR / "uploads"
BRAND_PHOTOS_DIR = UPLOAD_FOLDER / "brand_photos"

# Создание необходимых папок
os.makedirs(CLASSES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BRAND_PHOTOS_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================

def allowed_file(filename):
    """Проверяет допустимость расширения файла"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_table_name(name):
    """Очищает имя для использования в SQL"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

@contextmanager
def get_db():
    """Контекстный менеджер для работы с базой данных с повторными попытками"""
    retries = 0
    last_error = None
    
    while retries < MAX_RETRIES:
        try:
            conn = sqlite3.connect(
                DATABASE_PATH, 
                timeout=SQLITE_TIMEOUT,
                isolation_level=None
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            
            try:
                yield conn
                conn.commit()
                break
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()
                
        except sqlite3.OperationalError as e:
            last_error = e
            if "locked" in str(e):
                retries += 1
                logger.warning(f"БД заблокирована, попытка {retries}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY * retries)
                continue
            raise
        except Exception as e:
            last_error = e
            raise
    
    if retries == MAX_RETRIES:
        logger.error(f"Не удалось получить соединение после {MAX_RETRIES} попыток")
        raise sqlite3.OperationalError(f"Database is locked. Last error: {last_error}")

def check_database_integrity():
    """Проверяет целостность базы данных"""
    try:
        with get_db() as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result[0] == "ok":
                logger.info("Проверка целостности БД: OK")
                return True
            else:
                logger.error(f"Ошибка целостности БД: {result}")
                return False
    except Exception as e:
        logger.error(f"Ошибка при проверке целостности БД: {e}")
        return False

def init_db():
    """Инициализация базы данных"""
    try:
        with get_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alcohol_types (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        logger.info("База данных инициализирована")
        check_database_integrity()
    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}")
        raise

# ==============================================
# КЛАССЫ ДЛЯ РАБОТЫ С МОДЕЛЯМИ И ФОТО
# ==============================================

class ModelManager:
    """Класс для управления моделями ИИ"""
    def __init__(self):
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        """Загружает все модели при инициализации"""
        logger.info("Начало загрузки моделей ИИ...")
        
        model_files = list(MODELS_DIR.glob("*.keras"))
        if not model_files:
            raise ValueError(f"Не найдено моделей в {MODELS_DIR}")
        
        for model_file in model_files:
            alcohol_type = model_file.stem.lower()
            try:
                logger.info(f"Загрузка модели для {alcohol_type}...")
                self.models[alcohol_type] = load_model(model_file)
                
                input_shape = self.models[alcohol_type].input_shape
                output_shape = self.models[alcohol_type].output_shape
                logger.info(
                    f"Модель {alcohol_type} загружена. "
                    f"Вход: {input_shape}, Выход: {output_shape}"
                )
            except Exception as e:
                logger.error(f"Ошибка загрузки модели {alcohol_type}: {e}")
                raise

    def get_model(self, alcohol_type):
        """Возвращает модель по типу алкоголя"""
        model = self.models.get(alcohol_type.lower())
        if model is None:
            available = list(self.models.keys())
            raise ValueError(
                f"Модель для типа {alcohol_type} не найдена. "
                f"Доступные модели: {', '.join(available)}"
            )
        return model

class PhotoManager:
    """Класс для управления фотографиями"""
    @staticmethod
    def save_photo(file, brand, bottle_volume, remaining_volume):
        """Сохраняет фотографию с метаданными"""
        try:
            if not file or file.filename == '':
                raise ValueError("Неверный файл изображения")
            
            if not allowed_file(file.filename):
                raise ValueError("Недопустимый тип файла")
            
            if not brand or not isinstance(brand, str):
                raise ValueError("Неверное название бренда")
            
            try:
                bottle_volume = float(bottle_volume)
                remaining_volume = float(remaining_volume)
                if bottle_volume <= 0 or remaining_volume < 0:
                    raise ValueError("Объем должен быть положительным")
            except (ValueError, TypeError):
                raise ValueError("Неверный формат объема")

            safe_brand = re.sub(r'[^\w-]', '_', brand.strip())
            brand_dir = BRAND_PHOTOS_DIR / safe_brand
            os.makedirs(brand_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            
            new_filename = (
                f"{timestamp}_"
                f"{safe_brand}_"
                f"{int(bottle_volume)}ml_"
                f"{int(remaining_volume)}ml_"
                f"{name.replace(' ', '_')}{ext}"
            )
            
            file_path = brand_dir / new_filename
            file.save(file_path)
            
            logger.info(f"Фото сохранено: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Ошибка сохранения фото: {e}", exc_info=True)
            raise

# ==============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# ==============================================

def load_alcohol_classes():
    """Загружает классы алкоголя из файлов"""
    classes = {}
    expected_counts = {'gin': 60}
    
    try:
        logger.info(f"Загрузка классов из: {CLASSES_DIR}")
        
        if not CLASSES_DIR.exists():
            raise ValueError(f"Директория не найдена: {CLASSES_DIR}")
        
        for filename in os.listdir(CLASSES_DIR):
            if filename.endswith('.txt'):
                alcohol_type = filename[:-4].lower()
                filepath = CLASSES_DIR / filename
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    brands = []
                    for line in f:
                        brand = line.strip()
                        if brand:
                            brands.append(brand)
                    
                    if not brands:
                        logger.warning(f"Файл {filename} пуст")
                        continue
                    
                    classes[alcohol_type] = brands
                    count = len(brands)
                    
                    if alcohol_type in expected_counts and count != expected_counts[alcohol_type]:
                        logger.warning(
                            f"Для {alcohol_type} загружено {count} брендов, "
                            f"ожидалось {expected_counts[alcohol_type]}"
                        )
                    
                    logger.info(f"Загружено {count} брендов для {alcohol_type}")
        
        if not classes:
            raise ValueError("Не найдено классов алкоголя")
            
        return classes
        
    except Exception as e:
        logger.error(f"Ошибка загрузки классов: {e}", exc_info=True)
        raise

def get_or_create_table(alcohol_type):
    """Создает таблицу для типа алкоголя"""
    try:
        table_name = f"alc_{sanitize_table_name(alcohol_type)}"
        
        with get_db() as conn:
            cursor = conn.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            ''', (table_name,))
            
            if not cursor.fetchone():
                logger.info(f"Создаем таблицу {table_name}")
                
                conn.execute(f'''
                    CREATE TABLE "{table_name}" (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type_alcohol TEXT NOT NULL,
                        brand TEXT NOT NULL,
                        bottle_volume REAL NOT NULL,
                        bottle_weight REAL,
                        alcohol_weight_per_ml REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(brand, bottle_volume)
                    )
                ''')
                
                conn.execute(f'''
                    CREATE INDEX idx_{table_name}_brand 
                    ON "{table_name}" (brand)
                ''')
                
                conn.execute(f'''
                    CREATE INDEX idx_{table_name}_volume 
                    ON "{table_name}" (bottle_volume)
                ''')
                
                conn.execute('''
                    INSERT OR IGNORE INTO alcohol_types (name) VALUES (?)
                ''', (alcohol_type,))
                
                conn.commit()
                logger.info(f"Таблица {table_name} создана")
            else:
                logger.debug(f"Таблица {table_name} существует")
                
        return table_name
    except Exception as e:
        logger.error(f"Ошибка создания таблицы: {e}", exc_info=True)
        raise

def predict_with_filter(image_path, alcohol_type, model_manager, enabled_brands=None):
    """Предсказание бренда по изображению"""
    try:
        alcohol_type = alcohol_type.lower()
        logger.info(f"Начало предсказания для: {alcohol_type}")

        if alcohol_type not in ALCOHOL_CLASSES:
            available_types = list(ALCOHOL_CLASSES.keys())
            raise ValueError(f"Неподдерживаемый тип: {alcohol_type}. Доступно: {available_types}")

        all_brands = ALCOHOL_CLASSES[alcohol_type]

        if enabled_brands:
            enabled_brands = [b.lower().strip() for b in enabled_brands]
            brands = [b for b in all_brands if b.lower() in enabled_brands]
            if not brands:
                raise ValueError(f"Нет совпадений для брендов: {enabled_brands}")
            brand_indices = [i for i, b in enumerate(all_brands) if b.lower() in enabled_brands]
        else:
            brands = all_brands
            brand_indices = list(range(len(all_brands)))

        logger.debug(f"Используемые бренды ({len(brands)}): {brands}")

        model = model_manager.get_model(alcohol_type)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Не удалось прочитать изображение")

        img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_original = cv2.resize(img_original, (224, 224))
        img_original = img_original.astype('float32') / 255.0
        img_original = np.expand_dims(img_original, axis=0)

        cropped = tf.image.central_crop(img, 0.7)
        img_detail = tf.image.resize(cropped, (224, 224)).numpy()
        img_detail = cv2.cvtColor(img_detail, cv2.COLOR_BGR2RGB)
        img_detail = img_detail.astype('float32') / 255.0
        img_detail = np.expand_dims(img_detail, axis=0)

        model_input = [img_original, img_detail]
        predictions = model.predict(model_input, verbose=0)[0]

        filtered_predictions = predictions[brand_indices]
        filtered_predictions = filtered_predictions / np.sum(filtered_predictions)

        top_indices = np.argpartition(filtered_predictions, -2)[-2:]
        top_indices = top_indices[np.argsort(filtered_predictions[top_indices])][::-1]

        predicted_brand = brands[top_indices[0]]
        confidence = float(filtered_predictions[top_indices[0]])
        alternative_brand = brands[top_indices[1]]
        alternative_confidence = float(filtered_predictions[top_indices[1]])

        logger.info(
            f"Результат: {predicted_brand} ({confidence:.2%}), "
            f"альтернатива: {alternative_brand} ({alternative_confidence:.2%})"
        )
        
        return {
            "brand": predicted_brand,
            "confidence": confidence,
            "alternative_brands": [alternative_brand],
            "alternative_confidences": [alternative_confidence],
        }
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        raise ValueError(f"Ошибка при обработке: {str(e)}") from e

def ensure_class_folders():
    """Создает папки классов"""
    try:
        for alcohol_type in ALCOHOL_CLASSES.keys():
            learning_path = LEARNING_DIR / alcohol_type.capitalize() / "learning"
            os.makedirs(learning_path, exist_ok=True)
            
            class_file = CLASSES_DIR / f"{alcohol_type}.txt"
            if not class_file.exists():
                with open(class_file, 'w', encoding='utf-8') as f:
                    logger.info(f"Создан файл классов для {alcohol_type}")
                
                for brand in ALCOHOL_CLASSES[alcohol_type]:
                    brand_path = learning_path / brand
                    os.makedirs(brand_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Ошибка создания папок: {e}")
        raise

# ==============================================
# API ЭНДПОИНТЫ
# ==============================================

@app.route('/', methods=['GET'])
def index():
    """Основной эндпоинт"""
    return jsonify({
        "status": "работает",
        "доступные_типы": list(ALCOHOL_CLASSES.keys()),
        "версия": "1.2.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание бренда"""
    filepath = None
    try:
        if 'file' not in request.files:
            raise ValueError("Не предоставлен файл")
            
        file = request.files['file']
        if file.filename == '':
            raise ValueError("Пустой файл")
        
        alcohol_type = request.form.get('alcohol_type', '').strip().lower()
        if not alcohol_type:
            raise ValueError("Не указан тип алкоголя")
        
        logger.debug(f"Получен запрос: тип={alcohol_type}, файл={file.filename}")
        
        if alcohol_type not in ALCOHOL_CLASSES:
            available_types = list(ALCOHOL_CLASSES.keys())
            raise ValueError(f"Неподдерживаемый тип: '{alcohol_type}'. Доступно: {available_types}")

        enabled_brands = []
        if 'enabled_brands' in request.form:
            try:
                enabled_brands = json.loads(request.form['enabled_brands'])
                if not isinstance(enabled_brands, list):
                    raise ValueError("enabled_brands должен быть списком")
            except json.JSONDecodeError:
                raise ValueError("Неверный формат enabled_brands")
        
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        result = predict_with_filter(filepath, alcohol_type, app.model_manager, enabled_brands)
        
        return jsonify({
            "status": "success",
            "type": alcohol_type,
            **result
        })
    except Exception as e:
        logger.error(f"API ошибка: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "available_types": list(ALCOHOL_CLASSES.keys())
        }), 400
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/brands', methods=['GET'])
def get_brands():
    """Получение списка брендов"""
    try:
        alcohol_type = request.args.get('type', '').strip().lower()
        if not alcohol_type:
            raise ValueError("Не указан тип алкоголя")
            
        if alcohol_type not in ALCOHOL_CLASSES:
            raise ValueError(f"Неподдерживаемый тип: {alcohol_type}")
            
        return jsonify({
            "status": "success",
            "type": alcohol_type,
            "brands": ALCOHOL_CLASSES[alcohol_type]
        })
    except Exception as e:
        logger.error(f"Ошибка получения брендов: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/bottle', methods=['GET'])
def find_bottle():
    """Поиск бутылки"""
    try:
        alcohol_type = request.args.get('type', '').strip()
        brand = request.args.get('brand', '').strip()
        volume = request.args.get('volume', '0').strip()
        
        if not alcohol_type:
            raise ValueError("Тип алкоголя обязателен")
        if not brand:
            raise ValueError("Бренд обязателен")
        try:
            volume = float(volume)
            if volume <= 0:
                raise ValueError("Объем должен быть > 0")
        except ValueError:
            raise ValueError("Неверный формат объема")

        table_name = get_or_create_table(alcohol_type)
        
        with get_db() as conn:
            cursor = conn.execute(f'''
                SELECT * FROM "{table_name}"
                WHERE brand = ? AND bottle_volume = ?
            ''', (brand, volume))
            
            bottle = cursor.fetchone()
            
            if bottle:
                return jsonify({
                    "status": "found",
                    "data": dict(bottle)
                })
            
            return jsonify({
                "status": "not_found",
                "template": {
                    "type_alcohol": alcohol_type,
                    "brand": brand,
                    "bottle_volume": volume,
                    "required_fields": {
                        "bottle_weight": {
                            "type": "number",
                            "description": "Вес пустой бутылки в граммах",
                            "min": 0.1,
                            "max": 5000
                        }
                    },
                    "next_step": "calibration"
                }
            }), 404
    except ValueError as e:
        logger.error(f"Ошибка параметров: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/bottle', methods=['POST'])
def add_bottle():
    """Добавление бутылки"""
    try:
        data = request.get_json()
        
        required = ['type_alcohol', 'brand', 'bottle_volume', 'bottle_weight']
        if not all(field in data for field in required):
            missing = [f for f in required if f not in data]
            raise ValueError(f"Не хватает полей: {missing}")
        
        alcohol_type = data['type_alcohol'].strip()
        brand = data['brand'].strip()
        try:
            volume = float(data['bottle_volume'])
            weight = float(data['bottle_weight'])
            if volume <= 0 or weight <= 0:
                raise ValueError("Значения должны быть > 0")
        except (ValueError, TypeError):
            raise ValueError("Неверный формат числовых полей")

        table_name = get_or_create_table(alcohol_type)
        
        with get_db() as conn:
            cursor = conn.execute(f'''
                SELECT 1 FROM "{table_name}"
                WHERE brand = ? AND bottle_volume = ?
            ''', (brand, volume))
            
            if cursor.fetchone():
                return jsonify({
                    "error": "Бутылка уже существует"
                }), 400
            
            cursor = conn.execute(f'''
                INSERT INTO "{table_name}"
                (type_alcohol, brand, bottle_volume, bottle_weight)
                VALUES (?, ?, ?, ?)
            ''', (alcohol_type, brand, volume, weight))
            
            conn.commit()
            
            return jsonify({
                "status": "created",
                "bottle_id": cursor.lastrowid,
                "next_step": {
                    "action": "calibrate",
                    "message": "Выполните калибровку",
                    "required_fields": {
                        "added_ml": {
                            "type": "number",
                            "description": "Добавленный объем в мл",
                            "min": 1,
                            "max": 10000
                        },
                        "current_weight": {
                            "type": "number",
                            "description": "Текущий вес в граммах",
                            "min": 0.1,
                            "max": 20000
                        }
                    }
                }
            }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except sqlite3.IntegrityError:
        return jsonify({"error": "Ошибка базы данных"}), 500
    except Exception as e:
        logger.error(f"Ошибка добавления: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/bottle/<string:alcohol_type>/<int:bottle_id>', methods=['PUT'])
def calibrate_bottle(alcohol_type, bottle_id):
    """Калибровка бутылки"""
    try:
        data = request.get_json()
        
        required = ['added_ml', 'current_weight']
        if not all(field in data for field in required):
            missing = [f for f in required if f not in data]
            raise ValueError(f"Не хватает полей: {missing}")
        
        try:
            added_ml = float(data['added_ml'])
            current_weight = float(data['current_weight'])
            if added_ml <= 0 or current_weight <= 0:
                raise ValueError("Значения должны быть > 0")
        except (ValueError, TypeError):
            raise ValueError("Неверный формат числовых полей")

        table_name = get_or_create_table(alcohol_type)
        
        with get_db() as conn:
            cursor = conn.execute(f'''
                SELECT bottle_weight FROM "{table_name}"
                WHERE id = ?
            ''', (bottle_id,))
            
            bottle = cursor.fetchone()
            if not bottle:
                raise ValueError("Бутылка не найдена")
            
            empty_weight = bottle['bottle_weight']
            
            alcohol_weight = (current_weight - empty_weight) / added_ml
            alcohol_weight = round(alcohol_weight, 4)
            
            conn.execute(f'''
                UPDATE "{table_name}"
                SET alcohol_weight_per_ml = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (alcohol_weight, bottle_id))
            
            conn.commit()
            
            return jsonify({
                "status": "calibrated",
                "alcohol_weight_per_ml": alcohol_weight,
                "next_step": {
                    "action": "measure",
                    "message": "Теперь можно измерять остаток"
                }
            })
    except ValueError as e:
        logger.error(f"Ошибка в данных: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Ошибка калибровки: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/calculate', methods=['POST'])
def calculate_volume():
    """Расчет объема"""
    try:
        data = request.get_json()
        
        required = ['alcohol_type', 'bottle_id', 'current_weight']
        if not all(field in data for field in required):
            missing = [f for f in required if f not in data]
            raise ValueError(f"Не хватает полей: {missing}")
        
        alcohol_type = data['alcohol_type'].strip()
        bottle_id = data['bottle_id']
        current_weight = float(data['current_weight'])
        
        table_name = get_or_create_table(alcohol_type)
        
        with get_db() as conn:
            cursor = conn.execute(f'''
                SELECT bottle_weight, alcohol_weight_per_ml, bottle_volume 
                FROM "{table_name}"
                WHERE id = ?
            ''', (bottle_id,))
            
            bottle = cursor.fetchone()
            if not bottle:
                raise ValueError("Бутылка не найдена")
            
            if not bottle['alcohol_weight_per_ml']:
                raise ValueError("Бутылка не откалибрована")
            
            empty_weight = bottle['bottle_weight']
            weight_per_ml = bottle['alcohol_weight_per_ml']
            total_volume = bottle['bottle_volume']
            
            remaining_ml = (current_weight - empty_weight) / weight_per_ml
            remaining_ml = max(0, min(total_volume, remaining_ml))
            
            return jsonify({
                "status": "success",
                "remaining_ml": round(remaining_ml, 2),
                "bottle_volume": total_volume,
                "percentage": round((remaining_ml / total_volume) * 100, 1)
            })
    except ValueError as e:
        logger.error(f"Ошибка в данных: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Ошибка расчета: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/save_photo', methods=['POST'])
def save_photo():
    """Сохранение фотографии"""
    try:
        if 'file' not in request.files:
            raise ValueError("Не предоставлен файл")
            
        file = request.files['file']
        if file.filename == '':
            raise ValueError("Пустой файл")
        
        brand = request.form.get('brand', '').strip()
        if not brand:
            raise ValueError("Не указан бренд")
        
        try:
            bottle_volume = float(request.form.get('bottle_volume', '0'))
            remaining_volume = float(request.form.get('remaining_volume', '0'))
            if bottle_volume <= 0 or remaining_volume < 0:
                raise ValueError("Объем должен быть положительным")
        except ValueError:
            raise ValueError("Неверный формат объема")

        if not allowed_file(file.filename):
            raise ValueError("Недопустимый тип файла")

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        logger.debug(f"Получен файл: {file.filename}, размер: {file_size} bytes")

        safe_brand = re.sub(r'[^\w-]', '_', brand)
        brand_dir = BRAND_PHOTOS_DIR / safe_brand
        os.makedirs(brand_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        name, ext = os.path.splitext(original_filename)
        
        new_filename = (
            f"{timestamp}_"
            f"{safe_brand}_"
            f"{int(bottle_volume)}ml_"
            f"{int(remaining_volume)}ml_"
            f"{name.replace(' ', '_')}{ext}"
        )

        save_path = brand_dir / new_filename
        file.save(save_path)
        
        logger.info(f"Фото сохранено: {save_path}")
        
        return jsonify({
            "status": "success",
            "message": "Фото сохранено",
            "path": str(save_path),
            "brand": brand,
            "bottle_volume": bottle_volume,
            "remaining_volume": remaining_volume,
            "file_info": {
                "original_name": original_filename,
                "saved_name": new_filename,
                "size_bytes": file_size,
                "saved_at": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Ошибка сохранения фото: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

@app.route('/api/list_photos/<string:brand>', methods=['GET'])
def list_photos(brand):
    """Список фотографий"""
    try:
        safe_brand = re.sub(r'[^\w-]', '_', brand.strip())
        brand_dir = BRAND_PHOTOS_DIR / safe_brand
        
        if not os.path.exists(brand_dir):
            return jsonify({
                "status": "success",
                "message": "Папка не найдена",
                "photos": [],
                "count": 0
            })
        
        photos = []
        for filename in sorted(os.listdir(brand_dir), reverse=True):
            file_path = brand_dir / filename
            if os.path.isfile(file_path):
                parts = filename.split('_')
                try:
                    date_str = parts[0]
                    time_str = parts[1]
                    volume_str = parts[3]
                    remaining_str = parts[5]
                    
                    photos.append({
                        "filename": filename,
                        "path": str(file_path),
                        "size_bytes": os.path.getsize(file_path),
                        "metadata": {
                            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                            "time": f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
                            "volume_ml": int(volume_str.replace('ml', '')),
                            "remaining_ml": int(remaining_str.replace('ml', ''))
                        }
                    })
                except Exception as e:
                    logger.warning(f"Не удалось разобрать имя файла {filename}: {e}")
                    photos.append({
                        "filename": filename,
                        "path": str(file_path),
                        "size_bytes": os.path.getsize(file_path)
                    })
        
        return jsonify({
            "status": "success",
            "brand": brand,
            "photos": photos,
            "count": len(photos)
        })
    except Exception as e:
        logger.error(f"Ошибка получения списка фото: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==============================================
# ЗАПУСК СЕРВЕРА
# ==============================================

if __name__ == '__main__':
    try:
        logger.info("Инициализация сервера...")
        
        ALCOHOL_CLASSES = load_alcohol_classes()
        init_db()
        ensure_class_folders()
        
        app.model_manager = ModelManager()
        
        logger.info(f"Загружено {len(ALCOHOL_CLASSES)} типов алкоголя")
        for alc_type, brands in ALCOHOL_CLASSES.items():
            logger.info(f"  {alc_type}: {len(brands)} брендов")
        
        logger.info(f"Загружено {len(app.model_manager.models)} моделей ИИ")
        
        expected_models = ['gin', 'vodka', 'whiskey']
        for model in expected_models:
            if model not in app.model_manager.models:
                logger.warning(f"Ожидаемая модель {model} не загружена!")
        
        logger.info("Запуск сервера на порту 5000")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.critical(f"Ошибка запуска сервера: {e}", exc_info=True)