import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                    BatchNormalization, concatenate, Input,
                                    Conv2D, MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Настройки
BATCH_SIZE = 32
EPOCHS = 30
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 6
TRAIN_DIR = 'C:/my_project2/alcogole/Vermouth/learning'
VALID_DIR = 'C:/my_project2/alcogole/Vermouth/valid'

# Проверка доступности GPU
print("Доступные устройства:", tf.config.list_physical_devices())
if tf.config.list_physical_devices('GPU'):
    print("GPU будет использоваться для обучения")
    # Оптимизация для GPU
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("Обучение будет выполняться на CPU")

# 1. Функция для анализа цвета
def create_color_branch(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs, x)

# 2. Основная модель
def build_model():
    input_resnet = Input(shape=(*IMAGE_SIZE, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_resnet)
    
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    resnet_branch = GlobalAveragePooling2D()(base_model.output)
    resnet_branch = Dropout(0.3)(resnet_branch)
    
    input_color = Input(shape=(*IMAGE_SIZE, 3))
    color_branch = create_color_branch((*IMAGE_SIZE, 3))(input_color)
    color_branch = Dropout(0.3)(color_branch)
    
    combined = concatenate([resnet_branch, color_branch])
    x = Dense(1024, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=[input_resnet, input_color], outputs=predictions)
    return model

# 3. Генераторы данных (без multiprocessing)
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Функция для создания двух входов
    def dual_input_generator(generator):
        for x, y in generator:
            yield [x, x], y
    
    return dual_input_generator(train_generator), dual_input_generator(valid_generator), train_generator, valid_generator

# 4. Компиляция модели
def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    return model

# 5. Визуализация метрик
def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig('training_metrics.png')
    plt.close()

# 6. Обучение модели
def train_model():
    # Создаем генераторы
    train_gen, valid_gen, original_train_gen, original_valid_gen = create_data_generators()
    
    # Строим и компилируем модель
    model = build_model()
    model = compile_model(model)
    
    # Вычисляем шаги
    steps_per_epoch = original_train_gen.samples // BATCH_SIZE
    validation_steps = original_valid_gen.samples // BATCH_SIZE
    
    print(f"\nTraining samples: {original_train_gen.samples} ({steps_per_epoch} batches)")
    print(f"Validation samples: {original_valid_gen.samples} ({validation_steps} batches)\n")
    
    # Обучение (без multiprocessing)
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
    )
    
    # Сохранение и вывод результатов
    model.save('final_model.keras')
    plot_metrics(history)
    
    print("\nTraining completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    train_model()