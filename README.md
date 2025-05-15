# Студент группы М8О-407Б-21 Лютоев Илья Александрович

## Лабораторная работа №7
## Проведение исследований моделями семантической сегментации  

### 1. Выбор начальных условий  

#### a. Выбор набора данных  
**Датасет**: [Nails Segmentation](https://www.kaggle.com/datasets/vpapenko/nails-segmentation)  
**Обоснование**:  
- Практическая задача определения положения ногтей у человека  
- Бинарная сегментация: фон (0) и ноготь (1)
- 52 изображений с соответствующими масками  

#### b. Выбор метрик качества  
- **Accuracy**: Общая точность классификации пикселей  
- **Dice (mean Intersection over Union)**: Учет пересечения предсказаний и истинных масок

---

### 2. Создание бейзлайна и оценка качества  

#### a. Обучение моделей  
Использованы 2 архитектуры:  
1. **U-Net** с предобученным энкодером MobileNetV2  
2. **Кастомная CNN** (модифицированная U-Net архитектура)  

**Ключевые параметры обучения**:  
- Размер батча: 4
- Оптимизатор: Adam (lr=0.0001)
- Loss: Weighted CrossEntropy ([1.0, 4.0])  
- Эпохи: 25

#### b. Результаты базовых моделей  

| Модель          | Test Accuracy | Test Dice |
|-----------------|---------------|-----------|
| U-Net           | 0.9852        | 0.9174    |
| Custom Conv     | 0.9463        | 0.7793    |

---

### 3. Улучшение бейзлайна  

#### a. Гипотезы для улучшения:  
1. Использование разных энкодеров (ResNet, EfficientNet)  
2. Добавление аугментаций: цветовые искажения, случайные кадрирования  
3. Эксперименты с балансом классов (веса [1.0, 3.0])  
4. Применение Dice Loss вместо CrossEntropy

#### b-f. Реализация и сравнение  

**Ключевые модификации в коде**:  
```python
# Весовая функция для учета дисбаланса классов
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))
```
```python
class TwoConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = TwoConvLayers(in_channels=in_channels, out_channels=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        y = self.max_pool(x)
        return y, x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.block = TwoConvLayers(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, y):
        x = self.transpose(x)
        u = torch.cat([x, y], dim=1)
        u = self.block(u)
        return u

class Custom2Conv(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.enc_block1 = Encoder(in_channels=in_channels, out_channels=64)
        self.enc_block2 = Encoder(in_channels=64, out_channels=128)
        self.enc_block3 = Encoder(in_channels=128, out_channels=256)
        self.enc_block4 = Encoder(in_channels=256, out_channels=512)

        self.bottleneck = TwoConvLayers(in_channels=512, out_channels=1024)

        self.dec_block1 = Decoder(in_channels=1024, out_channels=512)
        self.dec_block2 = Decoder(in_channels=512, out_channels=256)
        self.dec_block3 = Decoder(in_channels=256, out_channels=128)
        self.dec_block4 = Decoder(in_channels=128, out_channels=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)
```

Ключевые наблюдения:

1. U-Net демонстрирует более стабильный рост метрик
2. Кастомная модель показывает худший рост рост точности в первых эпохах
3. U-Net раньше достигает плато 
4. Весовая функция помогла улучшить сегментацию ногтей

### 4. Выводы

- Эффективность U-Net: Лучшие показатели Dice (91.74% vs 77.93%)
- Малый размер батча (4) требует аккуратного подбора lr

---

## Лабораторная работа №8
## Проведение исследований моделями обнаружения и распознавания объектов  

### 1. Выбор начальных условий  

#### a. Выбор набора данных  
**Датасет**: [Wind Turbin Dataset](https://www.kaggle.com/code/ishangrotra/wind-turbines-object-detection)  
**Обоснование**:  
- Практическая задача навигации для дронов  
- Бинарная классификация: Ветряк (класс 0) и фон  

#### b. Выбор метрик качества  
- **Precision**: Точность детекций (доля верных срабатываний)  
- **Recall**: Полнота обнаружения объектов  
- **mAP50**: Средняя точность при IoU=0.5  
- **mAP50-95**: Усредненная точность для IoU от 0.5 до 0.95  

---

### 2. Создание бейзлайна и оценка качества  

#### a. Обучение модели  
Использована архитектура:  
- **YOLOv8n** (nano-версия) с предобученными весами  

**Ключевые параметры обучения**:  
- Размер батча: 8
- Размер изображения: 512×512
- Оптимизатор: Adam
- Эпохи: 50
- Устройство: CUDA

#### b. Результаты базовой модели  

| Метрика       | Значение  |
|---------------|-----------|
| Precision     | 0.6366089201329476 |
| Recall        | 0.5113773193483339 |
| mAP50         | 0.5209689105410606 |
| mAP50-95      | 0.19384053651952596 |

---

### 3. Улучшение бейзлайна  
#### a. Гипотезы для улучшения:
1. Использование улучшенных моделей YOLOv8m и YOLOv8x
2. Увеличение размера изображений
3. Применение трансферного обучения на доменных данных
4. Увеличение числа эпох

#### b-f. Реализация и сравнение  

**Ключевые наблюдения**:
1. Невысокий Precision 63% говорит о сложности входных данных
2. Низкий mAP50-95 показывает проблемы с локализацией
3. Recall в 51% показывает возможности улучшения в обнаружении

### 4. Выводы  
1. Эффективность YOLOv8: Модель демонстрирует недостаточно хорошую точность при высокой скорости обучения на GPU
2. Проблемы локализации: Разрыв между mAP50 и mAP50-95 требует улучшения регрессии боксов  
3. Оптимизация обучения:  
   - Необходимость увеличения эпох для сходимости  
