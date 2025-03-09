### 1. **Математическая модель нейрона. Примеры функций активации**  
Нейрон — базовая единица нейросети. Его математическая модель:  
\[ a = f\left( \sum_{i=1}^n w_i x_i + b \right), \]  
где \(x_i\) — входные данные, \(w_i\) — веса, \(b\) — смещение, \(f\) — функция активации.  

**Примеры функций активации**:  
- **Сигмоида**: \( f(x) = \frac{1}{1 + e^{-x}} \) (для бинарной классификации).  
- **ReLU**: \( f(x) = \max(0, x) \) (устраняет проблему затухающих градиентов).  
- **Softmax**: \( f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \) (для многоклассовой классификации).  
- **Tanh**: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) (нормализует выход в диапазон \([-1, 1]\)).

---

### 2. **Построение последовательной нейросетевой модели в Keras**  
Используется **Sequential API**, который позволяет добавлять слои последовательно:  
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100))  # Входной слой
model.add(Dense(1, activation='sigmoid'))  # Выходной слой
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

### 3. **Понятие и компоненты сверточной сети. Операция свертки и ее параметры**  
**Сверточная нейросеть (CNN)** используется для обработки изображений. Основные компоненты:  
- **Сверточные слои**: Применяют фильтры для выделения локальных признаков.  
- **Pooling**: Уменьшение размерности (например, MaxPooling).  
- **Полносвязные слои**: Для классификации.  

**Параметры свертки**:  
- **Размер ядра (kernel_size)**: Например, 3x3.  
- **Шаг (stride)**: Сдвиг фильтра (например, 1 или 2).  
- **Дополнение (padding)**: 'valid' (без дополнения) или 'same' (сохранение размера).  
- **Количество фильтров**: Число выходных каналов.

---

### 4. **Переобучение и слой Dropout**  
**Переобучение** возникает, когда модель запоминает тренировочные данные, но не обобщает.  
**Dropout** — метод регуляризации, случайно "отключающий" часть нейронов во время обучения.  
**Реализация в Keras**:  
```python
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 50% нейронов отключаются
```

---

### 5. **Перенос обучения (Transfer Learning)**  
Использование предобученной модели для новой задачи. **Способы**:  
- **Заморозка весов**: Обучение только новых слоев.  
- **Тонкая настройка (fine-tuning)**: Разморозка части слоев после начального обучения.  

**Пример**:  
```python
from keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Заморозка весов

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

### 6. **Архитектура CNN для бинарной классификации**  
Пример архитектуры:  
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Бинарный выход
])
```

---

### 7. **Архитектура CNN для множественной классификации**  
Изменения в выходном слое:  
```python
model.add(Dense(10, activation='softmax'))  # 10 классов
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

---

### 8. **Обнаружение объектов на изображении**  
Задача: Найти объекты и определить их класс + границы (bounding box).  
**Примеры моделей**:  
- **YOLO (You Only Look Once)**.  
- **Faster R-CNN**.  
- **SSD (Single Shot Detector)**.

---

### 9. **Сегментация изображений**  
**Виды**:  
- **Семантическая**: Классификация каждого пикселя (например, небо vs земля).  
- **Экземплярная**: Разделение объектов разных экземпляров (например, разные люди).  

**Примеры моделей**:  
- **U-Net** (для медицинской сегментации).  
- **Mask R-CNN** (экземплярная сегментация).

---

### 10. **Транспонированная свертка и апсемплинг**  
**Транспонированная свертка (Conv2DTranspose)** увеличивает размер изображения. Используется в генеративных моделях и декодерах.  
**Пример в Keras**:  
```python
model.add(Conv2DTranspose(64, (3,3), strides=2, padding='same'))
```

---

### 11. **Архитектура GAN (Generative Adversarial Network)**  
Состоит из двух сетей:  
- **Генератор**: Создает фейковые данные из шума.  
- **Дискриминатор**: Отличает реальные данные от фейковых.  

**Обучение**: Генератор стремится обмануть дискриминатор, а дискриминатор — правильно классифицировать данные.

---

### 12. **Энкодер и декодер**  
- **Энкодер**: Сжимает входные данные в скрытое представление (например, сверточные слои).  
- **Декодер**: Восстанавливает данные из скрытого представления (например, транспонированные свертки).  

**Применение**: Автоэнкодеры, генеративные модели, сегментация.

---

### 13. **Обучение и оценка модели**  
**Этапы**:  
1. Разделение данных: `train`, `validation`, `test`.  
2. Компиляция:  
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```  
3. Обучение:  
```python
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```  
4. Визуализация:  
```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
```

---

### 14. **Метрики качества для классификации**  
- **Accuracy**: Доля правильных ответов.  
- **Precision**: Точность (сколько из предсказанных положительных — верные).  
- **Recall**: Полнота (сколько реальных положительных найдено).  
- **F1-Score**: Гармоническое среднее precision и recall.  
- **ROC-AUC**: Площадь под кривой ошибок.

---

### 15. **Метрики для сегментации**  
- **Intersection over Union (IoU)**: \(\frac{TP}{TP + FP + FN}\).  
- **Dice Coefficient**: \(\frac{2 \cdot TP}{2 \cdot TP + FP + FN}\).  
- **Pixel Accuracy**: Доля правильно классифицированных пикселей.

---

### 16. **Автоэнкодер**  
Архитектура:  
```python
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2,2))(x)

x = Conv2DTranspose(32, (3,3), activation='relu', padding='same')(encoded)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
```

---

### 17. **Функциональный API в Keras**  
Позволяет создавать сложные архитектуры (не только последовательные):  
```python
from keras.layers import Input, Dense, concatenate
from keras.models import Model

input1 = Input(shape=(64,))
input2 = Input(shape=(128,))
x = Dense(32, activation='relu')(input1)
y = Dense(64, activation='relu')(input2)
combined = concatenate([x, y])
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[input1, input2], outputs=output)
```

---

### 18. **Подготовка данных для изображений**  
- **Нормализация**: Приведение значений пикселей к диапазону [0, 1].  
- **Аугментация**: Повороты, сдвиги, изменение яркости (используйте `ImageDataGenerator`).  
- **Разделение данных**: Обычно 60-20-20 (train-val-test).  

**Пример аугментации**:  
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    horizontal_flip=True
)
```

---

### 19. **Расширение данных (Data Augmentation)**  
Увеличение разнообразия данных без сбора новых:  
- **Геометрические преобразования**: Поворот, отражение, масштабирование.  
- **Цветовые искажения**: Изменение яркости, контраста.  
- **Шум**: Добавление гауссова шума.

---

### 20. **Использование предобученной модели в Keras**  
Пример с **MobileNet** для классификации:  
```python
from keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Заморозка весов

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

---

Этот материал охватывает ключевые концепции нейронных сетей, их реализации в Keras и примеры применения в различных задачах. Для углубленного изучения рекомендуется экспериментировать с кодом и изучать документацию Keras.
