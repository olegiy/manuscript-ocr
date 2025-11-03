# Визуализация примеров в TensorBoard

## Что добавлено

Во время обучения TRBA распознавателя автоматически визуализируются 10 случайных примеров распознавания на каждой валидационной эпохе **в TensorBoard**.

## Как это работает

### Что показывается

На каждой эпохе валидации в TensorBoard появляется сетка изображений, где:

- **Изображение**: входное изображение текста
- **GT (Ground Truth)**: правильный текст (зеленым если совпал, красным если ошибка)
- **Pred (Prediction)**: что распознала модель (синим цветом)

### Где смотреть

1. Запустите обучение:
   ```python
   from manuscript.recognizers import TRBA
   
   TRBA.train(
       train_csvs="data/train.csv",
       train_roots="data/images",
       val_csvs="data/val.csv",
       val_roots="data/val_images",
       exp_dir="experiments/my_exp",
       dual_validate=True,  # Покажет greedy + beam примеры
       epochs=100,
   )
   ```

2. Откройте TensorBoard:
   ```bash
   tensorboard --logdir experiments/my_exp/logs
   ```

3. Перейдите во вкладку **IMAGES**

4. Вы увидите:
   - `Predictions/greedy` - примеры с greedy декодированием
   - `Predictions/beam` - примеры с beam search (если `dual_validate=True`)

### Пример визуализации

В TensorBoard вы увидите сетку 5x2 (или другую, в зависимости от количества примеров):

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ [Image 1]   │ [Image 2]   │ [Image 3]   │ [Image 4]   │ [Image 5]   │
│ GT: Привет  │ GT: Мир     │ GT: Тест    │ GT: Модель  │ GT: Данные  │
│ Pred:Привет │ Pred: Мир   │ Pred: Тест  │ Pred:Модель │ Pred:Данные │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ [Image 6]   │ [Image 7]   │ [Image 8]   │ [Image 9]   │ [Image 10]  │
│ GT: Текст   │ GT: Слово   │ GT: Буква   │ GT: Строка  │ GT: Символ  │
│ Pred: Текст │ Pred: Слово │ Pred: Букаа │ Pred:Строка │ Pred:Символ │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

Зеленый GT = правильное распознавание  
Красный GT = ошибка распознавания  
Синий Pred = предсказание модели

## Преимущества

✅ **Визуальная оценка**: сразу видно качество на реальных примерах  
✅ **Типичные ошибки**: легко заметить, что модель путает  
✅ **Динамика обучения**: можно проследить, как улучшается качество по эпохам  
✅ **Случайные примеры**: каждый раз новые, полный охват датасета  
✅ **Два режима**: greedy (быстро) и beam search (точнее)  

## Настройка

### Изменить количество примеров

В файле `src/manuscript/recognizers/_trba/training/train.py` найдите:

```python
visualize_predictions_tensorboard(
    ...
    num_samples=10,  # Измените на нужное число
    ...
)
```

### Отключить визуализацию

Закомментируйте блок с `visualize_predictions_tensorboard()` в файле `train.py`.

### Изменить размер сетки

В функции `visualize_predictions_tensorboard` найдите:

```python
grid = torchvision.utils.make_grid(
    images_with_text,
    nrow=min(5, len(images_with_text)),  # Измените 5 на нужное число
    ...
)
```

## Технические детали

- **Файл**: `src/manuscript/recognizers/_trba/training/train.py`
- **Функция**: `visualize_predictions_tensorboard()`
- **Вызов**: автоматически после валидации
- **Формат**: PNG изображения в TensorBoard
- **Размер**: изображение + 60px для текста снизу
- **Цвета**: 
  - GT зеленый (0, 128, 0) если правильно
  - GT красный (255, 0, 0) если ошибка
  - Pred синий (0, 0, 255)

## Требования

- `torchvision` (уже установлено)
- `PIL` / `Pillow` (уже установлено)
- `tensorboard` (уже установлено)

## Пример использования

```python
from manuscript.recognizers import TRBA

# Обучение с визуализацией в TensorBoard
best_model = TRBA.train(
    train_csvs="data/train.csv",
    train_roots="data/images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="experiments/trba_exp1",
    dual_validate=True,  # Покажет greedy + beam
    epochs=100,
)

# Откройте TensorBoard:
# tensorboard --logdir experiments/trba_exp1/logs
# Затем перейдите на http://localhost:6006
# Вкладка IMAGES → Predictions/greedy и Predictions/beam
```

## Отличие от консольного вывода

**Старый подход (консоль)**:
- Только текст
- Не видно сами изображения
- Сложно оценить визуально

**Новый подход (TensorBoard)**:
- Видны изображения + текст
- Легко оценить качество визуально
- История по всем эпохам
- Красивые сетки изображений

## Troubleshooting

**Не вижу изображения в TensorBoard?**
- Убедитесь, что TensorBoard запущен с правильным --logdir
- Проверьте, что прошла хотя бы одна эпоха валидации
- Обновите страницу в браузере (F5)

**Текст не читается?**
- Попробуйте увеличить `text_height` в функции (сейчас 60px)
- Проверьте установку шрифта Arial (или используется дефолтный)

**Слишком много/мало примеров?**
- Измените параметр `num_samples` в вызовах функции
