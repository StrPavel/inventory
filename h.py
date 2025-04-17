import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                    BatchNormalization, concatenate, Input,
                                    Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import os

# Настройки
BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 60
TRAIN_DIR = 'C:/my_project2/alcogole/Gin/learning'
VALID_DIR = 'C:/my_project2/alcogole/Gin/valid'

# 1. Функция для анализа цвета
def create_color_branch(input_shape):
    """Создает ветку для анализа цвета"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs, x)

# 2. Основная модель с ResNet50 + цветом
def build_model():
    """Модель с двумя входами: ResNet50 (форма/текстура) + цвет"""
    # Вход для ResNet50
    input_resnet = Input(shape=(*IMAGE_SIZE, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_resnet)
    
    # Замораживаем часть слоев
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Ветка ResNet
    resnet_branch = GlobalAveragePooling2D()(base_model.output)
    
    # Ветка цвета (использует уменьшенное изображение для акцента на цвете)
    input_color = Input(shape=(*IMAGE_SIZE, 3))
    color_branch = create_color_branch((*IMAGE_SIZE, 3))(input_color)
    
    # Объединение веток
    combined = concatenate([resnet_branch, color_branch])
    x = Dense(1024, activation='relu')(combined)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=[input_resnet, input_color], outputs=predictions)
    return model

# 3. Генераторы данных с аккуратной аугментацией цвета
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=[0.8, 1.2],  # Только яркость (не искажаем оттенки)
        horizontal_flip=True
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Для валидации - без аугментации
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Генератор с двумя входами (одинаковые изображения для ResNet и цвета)
    def dual_input_generator(generator):
        while True:
            x, y = next(generator)
            yield [x, x], y
    
    return dual_input_generator(train_generator), dual_input_generator(valid_generator)

# 4. Компиляция с кастомными метриками цвета
def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3),
            # Добавляем метрику для контроля цветовых признаков
            tf.keras.metrics.CategoricalAccuracy(name='color_accuracy')
        ]
    )
    return model

# Остальные функции (обучение, оценка) остаются без изменений

def train_model():
    train_gen, valid_gen = create_data_generators()
    model = build_model()
    model = compile_model(model)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.batch_size,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('best_color_model.keras', save_best_only=True)
        ]
    )
    
    # Визуализация и оценка
    plot_metrics(history)
    evaluate_model(model, valid_gen)

def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['color_accuracy'], label='Color Train Accuracy')
    plt.plot(history.history['val_color_accuracy'], label='Color Val Accuracy')
    plt.title('Color Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['top3_accuracy'], label='Top-3 Train Accuracy')
    plt.plot(history.history['val_top3_accuracy'], label='Top-3 Val Accuracy')
    plt.title('Shape/Texture Accuracy')
    plt.legend()
    plt.savefig('color_metrics.png')

if __name__ == "__main__":
    train_model()