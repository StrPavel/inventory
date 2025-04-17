import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import datetime

# Настройки
BATCH_SIZE = 32
EPOCHS = 12
NUM_CLASSES = 60
IMAGE_SIZE = (256, 256)
TRAIN_DIR = 'C:/my_project2/learning'
VALID_DIR = 'C:/my_project2/valid'

# 1. Кастомная аугментация
def apply_color_augmentation(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if image.shape[-1] == 3:
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.1)
    return image

# 2. Генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    preprocessing_function=apply_color_augmentation
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

# 3. Загрузка модели
model = load_model('final_color_augmented_model.keras')

# 4. Стратегия разморозки (главные изменения здесь!)
# Проверяем текущие trainable слои
print("Текущие размороженные слои:")
for layer in model.layers:
    if layer.trainable:
        print(f"- {layer.name}")

# Добавляем L2-регуляризацию для новых слоёв
l2_reg = tf.keras.regularizers.l2(0.0005)

# Размораживаем:
# 1) Сохраняем предыдущие размороженные слои [-25:-20]
# 2) Добавляем 3 новых слоя [-28:-25]
for layer in model.layers[-28:-20]:
    layer.trainable = True
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = l2_reg
    print(f"Разморожен слой: {layer.name}")

# 5. Компиляция с кастомизированным LR
optimizer = Adam(
    learning_rate=0.00003,  # Среднее значение между 1e-4 и 1e-5
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]  # Добавляем Top-5 accuracy
)

# 6. Улучшенные коллбеки
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(
        monitor='val_top_k_categorical_accuracy',  # Мониторим Top-5 accuracy
        patience=10,
        mode='max',
        restore_best_weights=True,
        min_delta=0.0005
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'enhanced_model.keras',
        save_best_only=True,
        monitor='val_top_k_categorical_accuracy',
        mode='max'
    ),
    TensorBoard(
        log_dir=log_dir,
        profile_batch=0  # Отключаем профилирование для стабильности
    )
]

# 7. Обучение с прогресс-баром
print("\nНачинаем обучение...")
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    validation_data=valid_generator,
    validation_steps=max(1, valid_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 8. Расширенная визуализация
metrics = ['accuracy', 'top_k_categorical_accuracy', 'loss']
plt.figure(figsize=(18, 12))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'Model {metric.capitalize()}', pad=10)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend()

plt.tight_layout()
plt.savefig('enhanced_training_metrics.png', dpi=300)
plt.show()

# 9. Детальная оценка с дополнительными метриками
print("\nРасширенный отчет:")
Y_pred = model.predict(valid_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Добавляем вывод confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(valid_generator.classes, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=valid_generator.class_indices.keys())
disp.plot(xticks_rotation=90)
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=200)

print(classification_report(
    valid_generator.classes,
    y_pred,
    target_names=list(valid_generator.class_indices.keys()),
    digits=4
))

# 10. Сохранение с дополнительной информацией
model.save('final_enhanced_model.keras')
print("\nМодель успешно сохранена как 'final_enhanced_model.keras'")

# Дополнительно: сохраняем архитектуру
with open('model_architecture.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))