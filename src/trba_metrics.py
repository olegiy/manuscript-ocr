import os
import time
import csv
from collections import Counter, defaultdict
from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import (
    compute_cer,
    compute_wer,
    compute_accuracy,
)
from tqdm import tqdm

# Levenshtein only for detailed error analysis
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("Warning: python-Levenshtein not installed. Detailed error analysis will be limited.")


# Wrapper functions for single-item compatibility
def character_error_rate(reference: str, hypothesis: str) -> float:
    """Single-item CER for compatibility."""
    return compute_cer([reference], [hypothesis])


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Single-item WER for compatibility."""
    return compute_wer([reference], [hypothesis])


# === Пути ===
# Пути к модели и данным
WEIGHTS_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth"
CONFIG_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
CHARSET_PATH = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

# Датасеты для валидации
datasets = [
    {
        "image_dir": r"C:\shared\orig_cyrillic\test",
        "gt_path": r"C:\shared\orig_cyrillic\test.csv",
    },
    {
        "image_dir": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val",
        "gt_path": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val_converted.csv",
    },
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\labels.csv",
    },
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv",
    },
]


model_path = r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx"
config_path = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"

batch_size = 64

# === Читаем GT-файлы из всех датасетов ===
gt_data = {}
total_gt_lines = 0

for idx, dataset in enumerate(datasets, 1):
    image_dir = dataset["image_dir"]
    gt_path = dataset["gt_path"]
    
    print(f"📂 Датасет {idx}: {os.path.basename(image_dir)}")
    
    dataset_gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                fname = row[0].strip()
                text = ",".join(row[1:]).strip()  # На случай если в тексте есть запятые
                dataset_gt[fname] = text
    
    print(f"   Загружено {len(dataset_gt)} записей из {os.path.basename(gt_path)}")
    total_gt_lines += len(dataset_gt)
    
    # Добавляем в общий словарь с проверкой на дубликаты
    for fname, text in dataset_gt.items():
        if fname in gt_data:
            print(f"   ⚠️  Дубликат файла: {fname} (будет использована последняя версия)")
        gt_data[fname] = text

print(f"\n📄 Всего загружено {total_gt_lines} записей из {len(datasets)} датасетов(а)")
print(f"📄 Уникальных файлов: {len(gt_data)}")

# === Сканируем изображения из всех датасетов ===
valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images = []

for idx, dataset in enumerate(datasets, 1):
    image_dir = dataset["image_dir"]
    
    dataset_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    
    if not dataset_images:
        print(f"⚠️  Датасет {idx}: В папке {image_dir} не найдено изображений!")
    else:
        print(f"📁 Датасет {idx}: Найдено {len(dataset_images)} изображений")
        images.extend(dataset_images)

# === Ограничиваем количество изображений ===
max_images = 100000000000000
if len(images) > max_images:
    print(f"⚠️ Берём только первые {max_images} изображений из {len(images)}")
    images = images[:max_images]

if not images:
    raise RuntimeError(f"❌ Не найдено изображений ни в одном датасете!")

print(f"\n📁 ИТОГО: {len(images)} изображений для распознавания")

# === Инициализация модели ===
recognizer = TRBA(weights_path=model_path, config_path=config_path)

# === Выбор режима декодирования ===
# Доступные режимы: "greedy", "beam"

# === Распознаём ===
start_time = time.perf_counter()
results = recognizer.predict(images=images, batch_size=batch_size)
end_time = time.perf_counter()
print(results)
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

# ============================================
# ФУНКЦИЯ ДЛЯ ФИЛЬТРАЦИИ ТОЛЬКО БУКВ И ЦИФР
# ============================================

def filter_chars_only(text):
    """
    Оставляет только буквы (включая кириллицу и дореформенные) и цифры.
    Убирает пробелы, пунктуацию и спецсимволы.
    """
    # Определяем допустимые символы: буквы латиницы, кириллицы (включая дореформенные) и цифры
    allowed_chars = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        'ѣѢіІѳѲѵѴѫѪѭѬѯѮѱѰѡѠѕѕѧѦѩѨ'  # Дореформенные символы
        '0123456789'
    )
    return ''.join(c for c in text if c in allowed_chars)

# === Сопоставляем с ground truth ===
refs, hyps = [], []
total_cer, total_wer = 0.0, 0.0
cer_count, wer_count = 0, 0
error_details = []  # Для детального анализа ошибок

# Создаем уникальные имена для датасетов и нормализуем пути
dataset_mapping = {}  # путь к изображению -> уникальное имя датасета
dataset_results = {}  # уникальное имя -> метрики

for idx, dataset in enumerate(datasets, 1):
    # Создаем уникальное имя: если базовое имя не уникально, добавляем часть родительского пути
    base_name = os.path.basename(dataset["image_dir"])
    
    # Проверяем уникальность базового имени
    existing_names = [name for name in dataset_results.keys()]
    if base_name in existing_names:
        # Добавляем часть пути для уникальности
        parent = os.path.basename(os.path.dirname(dataset["image_dir"]))
        dataset_name = f"{parent}_{base_name}"
    else:
        dataset_name = base_name
    
    # Нормализуем путь к директории для точного сопоставления
    normalized_dir = os.path.normpath(dataset["image_dir"])
    dataset_mapping[normalized_dir] = dataset_name
    
    dataset_results[dataset_name] = {
        'refs': [],
        'hyps': [],
        'total_cer': 0.0,
        'total_wer': 0.0,
        'count': 0
    }

print("\n=== Результаты распознавания ===")
for path, result in zip(images, results):
    pred_text = result["text"]
    score = result["confidence"]
    fname = os.path.basename(path)
    ref_text = gt_data.get(fname)
    if ref_text is None:
        print(f"{fname:40s} → {pred_text:20s} (no GT)")
        continue

    refs.append(ref_text)
    hyps.append(pred_text)

    cer = character_error_rate(ref_text, pred_text)
    wer = word_error_rate(ref_text, pred_text)

    total_cer += cer
    total_wer += wer
    cer_count += 1
    wer_count += 1
    
    # Определяем к какому датасету относится изображение
    # Нормализуем путь изображения и ищем точное совпадение директории
    normalized_path = os.path.normpath(path)
    path_dir = os.path.dirname(normalized_path)
    
    dataset_name = dataset_mapping.get(path_dir)
    if dataset_name:
        dataset_results[dataset_name]['refs'].append(ref_text)
        dataset_results[dataset_name]['hyps'].append(pred_text)
        dataset_results[dataset_name]['total_cer'] += cer
        dataset_results[dataset_name]['total_wer'] += wer
        dataset_results[dataset_name]['count'] += 1
    
    # Сохраняем детали для анализа
    if ref_text != pred_text:
        error_details.append({
            'fname': fname,
            'ref': ref_text,
            'hyp': pred_text,
            'cer': cer,
            'wer': wer,
            'confidence': score
        })

    print(
        f"{fname:40s} → {pred_text:20s} | GT: {ref_text:20s} | CER={cer:.3f} | WER={wer:.3f}"
    )

# === Метрики ===
acc = compute_accuracy(refs, hyps)

# Регистронезависимая точность (case-insensitive)
acc_case_insensitive = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower()) / max(len(refs), 1)

# Точность только по буквам и цифрам (chars only - без спецсимволов и пробелов)
refs_chars_only = [filter_chars_only(r) for r in refs]
hyps_chars_only = [filter_chars_only(h) for h in hyps]
acc_chars_only = sum(1 for r, h in zip(refs_chars_only, hyps_chars_only) if r.lower() == h.lower()) / max(len(refs), 1)

# Точность с учетом только регистра (когда все символы верны, но регистр другой)
case_only_errors = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower() and r != h)

avg_cer = total_cer / max(cer_count, 1)
avg_wer = total_wer / max(wer_count, 1)

# CER/WER для chars only (пропускаем пары с пустыми строками)
total_cer_chars_only = 0.0
total_wer_chars_only = 0.0
chars_only_count = 0

for r, h in zip(refs_chars_only, hyps_chars_only):
    # Пропускаем пары где хотя бы одна строка пустая
    if not r and not h:
        # Обе пустые - считаем как совпадение (CER=0, WER=0)
        chars_only_count += 1
        continue
    elif not r or not h:
        # Одна пустая, другая нет - считаем как полную ошибку
        total_cer_chars_only += 1.0
        total_wer_chars_only += 1.0
        chars_only_count += 1
        continue
    
    # Обе непустые - считаем нормально
    total_cer_chars_only += character_error_rate(r, h)
    total_wer_chars_only += word_error_rate(r, h)
    chars_only_count += 1

avg_cer_chars_only = total_cer_chars_only / max(chars_only_count, 1)
avg_wer_chars_only = total_wer_chars_only / max(chars_only_count, 1)

print("\n=== Сводка ===")
print(f"Accuracy (case-sensitive):     {acc*100:.2f}%")
print(f"Accuracy (case-insensitive):   {acc_case_insensitive*100:.2f}%")
print(f"Accuracy (chars only):         {acc_chars_only*100:.2f}%")
print(f"Case-only errors:              {case_only_errors} ({case_only_errors/max(len(refs), 1)*100:.2f}%)")
print(f"Avg CER:  {avg_cer:.4f}")
print(f"Avg WER:  {avg_wer:.4f}")
print(f"Processed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")

# ============================================
# ТАБЛИЦА МЕТРИК ПО ДАТАСЕТАМ
# ============================================

print("\n" + "="*100)
print("📊 МЕТРИКИ ПО ДАТАСЕТАМ")
print("="*100)

# Подготовка данных для таблицы
metrics_table = []

for dataset_name, data in dataset_results.items():
    if data['count'] == 0:
        continue
    
    # Accuracy (case-sensitive)
    acc_ds = compute_accuracy(data['refs'], data['hyps'])
    
    # Accuracy (case-insensitive)
    acc_ci_ds = sum(1 for r, h in zip(data['refs'], data['hyps']) if r.lower() == h.lower()) / max(data['count'], 1)
    
    # Accuracy (chars only)
    refs_co_ds = [filter_chars_only(r) for r in data['refs']]
    hyps_co_ds = [filter_chars_only(h) for h in data['hyps']]
    acc_co_ds = sum(1 for r, h in zip(refs_co_ds, hyps_co_ds) if r.lower() == h.lower()) / max(data['count'], 1)
    
    # CER/WER
    avg_cer_ds = data['total_cer'] / data['count']
    avg_wer_ds = data['total_wer'] / data['count']
    
    # CER/WER (case-insensitive)
    cer_ci_ds = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(data['refs'], data['hyps'])) / data['count']
    wer_ci_ds = sum(word_error_rate(r.lower(), h.lower()) for r, h in zip(data['refs'], data['hyps'])) / data['count']
    
    # CER/WER (chars only) - с обработкой пустых строк
    total_cer_co_ds = 0.0
    total_wer_co_ds = 0.0
    co_count_ds = 0
    
    for r, h in zip(refs_co_ds, hyps_co_ds):
        if not r and not h:
            co_count_ds += 1
            continue
        elif not r or not h:
            total_cer_co_ds += 1.0
            total_wer_co_ds += 1.0
            co_count_ds += 1
            continue
        total_cer_co_ds += character_error_rate(r, h)
        total_wer_co_ds += word_error_rate(r, h)
        co_count_ds += 1
    
    cer_co_ds = total_cer_co_ds / max(co_count_ds, 1)
    wer_co_ds = total_wer_co_ds / max(co_count_ds, 1)
    
    metrics_table.append({
        'Dataset': dataset_name,
        'Count': data['count'],
        'Acc (CS)': acc_ds,
        'Acc (CI)': acc_ci_ds,
        'Acc (CO)': acc_co_ds,
        'CER (CS)': avg_cer_ds,
        'CER (CI)': cer_ci_ds,
        'CER (CO)': cer_co_ds,
        'WER (CS)': avg_wer_ds,
        'WER (CI)': wer_ci_ds,
        'WER (CO)': wer_co_ds,
    })

# Добавляем общую строку (TOTAL)
# Case-insensitive метрики для всех данных
total_cer_ci = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps)) / max(len(refs), 1)
total_wer_ci = sum(word_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps)) / max(len(refs), 1)

metrics_table.append({
    'Dataset': 'TOTAL',
    'Count': len(refs),
    'Acc (CS)': acc,
    'Acc (CI)': acc_case_insensitive,
    'Acc (CO)': acc_chars_only,
    'CER (CS)': avg_cer,
    'CER (CI)': total_cer_ci,
    'CER (CO)': avg_cer_chars_only,
    'WER (CS)': avg_wer,
    'WER (CI)': total_wer_ci,
    'WER (CO)': avg_wer_chars_only,
})

# Вывод таблицы
print(f"\n{'Dataset':<30} {'Count':>6} {'Acc(CS)':>8} {'Acc(CI)':>8} {'Acc(CO)':>8} {'CER(CS)':>8} {'CER(CI)':>8} {'CER(CO)':>8} {'WER(CS)':>8} {'WER(CI)':>8} {'WER(CO)':>8}")
print("-" * 100)
for row in metrics_table:
    is_total = row['Dataset'] == 'TOTAL'
    sep = "=" if is_total else "-"
    if is_total:
        print(sep * 100)
    print(f"{row['Dataset']:<30} {row['Count']:>6} {row['Acc (CS)']*100:>7.2f}% {row['Acc (CI)']*100:>7.2f}% {row['Acc (CO)']*100:>7.2f}% "
          f"{row['CER (CS)']:>8.4f} {row['CER (CI)']:>8.4f} {row['CER (CO)']:>8.4f} "
          f"{row['WER (CS)']:>8.4f} {row['WER (CI)']:>8.4f} {row['WER (CO)']:>8.4f}")

# Сохранение таблицы в CSV
csv_output_path = os.path.join(os.path.dirname(model_path), "metrics_by_dataset.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'Dataset', 'Count', 
        'Acc (CS)', 'Acc (CI)', 'Acc (CO)',
        'CER (CS)', 'CER (CI)', 'CER (CO)',
        'WER (CS)', 'WER (CI)', 'WER (CO)'
    ])
    writer.writeheader()
    for row in metrics_table:
        writer.writerow(row)

print(f"\n💾 Таблица метрик сохранена: {csv_output_path}")
print("="*100)

# ============================================
# ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК
# ============================================

def analyze_character_errors(refs, hyps):
    """Анализ ошибок на уровне символов"""
    
    substitutions = Counter()  # (правильный, ошибочный)
    insertions = Counter()     # вставленный символ
    deletions = Counter()      # удалённый символ
    
    error_positions = {'start': 0, 'middle': 0, 'end': 0}  # Позиция ошибок
    
    for ref, hyp in zip(refs, hyps):
        if ref == hyp:
            continue
            
        # Используем операции Levenshtein для детального анализа
        ops = Levenshtein.editops(ref, hyp)
        
        for op_type, ref_pos, hyp_pos in ops:
            # Определяем позицию ошибки в слове
            word_len = len(ref)
            if ref_pos < word_len * 0.2:
                error_positions['start'] += 1
            elif ref_pos > word_len * 0.8:
                error_positions['end'] += 1
            else:
                error_positions['middle'] += 1
            
            if op_type == 'replace':
                ref_char = ref[ref_pos] if ref_pos < len(ref) else ''
                hyp_char = hyp[hyp_pos] if hyp_pos < len(hyp) else ''
                substitutions[(ref_char, hyp_char)] += 1
            elif op_type == 'insert':
                hyp_char = hyp[hyp_pos] if hyp_pos < len(hyp) else ''
                insertions[hyp_char] += 1
            elif op_type == 'delete':
                ref_char = ref[ref_pos] if ref_pos < len(ref) else ''
                deletions[ref_char] += 1
    
    return substitutions, insertions, deletions, error_positions


def analyze_word_lengths(error_details):
    """Анализ длины слов с ошибками"""
    error_lengths = []
    correct_lengths = []
    
    for detail in error_details:
        error_lengths.append(len(detail['ref']))
    
    return error_lengths


def analyze_error_types(error_details):
    """Анализ типов ошибок"""
    
    total_errors = len(error_details)
    
    # Классификация ошибок
    length_mismatch = 0
    case_errors = 0
    similar_chars = 0
    completely_wrong = 0
    
    for detail in error_details:
        ref = detail['ref']
        hyp = detail['hyp']
        
        if len(ref) != len(hyp):
            length_mismatch += 1
        elif ref.lower() == hyp.lower():
            case_errors += 1
        else:
            # Проверяем схожесть
            distance = Levenshtein.distance(ref, hyp)
            if distance <= 2:
                similar_chars += 1
            else:
                completely_wrong += 1
    
    return {
        'total': total_errors,
        'length_mismatch': length_mismatch,
        'case_errors': case_errors,
        'similar_chars': similar_chars,
        'completely_wrong': completely_wrong
    }


if error_details:
    print("\n" + "="*80)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК")
    print("="*80)
    
    # 1. Общая статистика ошибок
    print(f"\n1️⃣ ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего примеров: {len(refs)}")
    print(f"   Правильных: {len(refs) - len(error_details)}")
    print(f"   С ошибками: {len(error_details)} ({len(error_details)/len(refs)*100:.1f}%)")
    
    # Регистронезависимые метрики (уже посчитаны выше)
    
    print(f"\n   📏 Метрики (case-sensitive):")
    print(f"      Accuracy: {acc*100:.2f}%")
    print(f"      CER: {avg_cer:.4f}")
    print(f"      WER: {avg_wer:.4f}")
    
    print(f"\n   📏 Метрики (case-insensitive):")
    print(f"      Accuracy: {acc_case_insensitive*100:.2f}%")
    print(f"      CER: {total_cer_ci:.4f}")
    print(f"      WER: {total_wer_ci:.4f}")
    if avg_cer > 0:
        print(f"      Улучшение CER: {(avg_cer - total_cer_ci)/avg_cer*100:.1f}%")
    if avg_wer > 0:
        print(f"      Улучшение WER: {(avg_wer - total_wer_ci)/avg_wer*100:.1f}%")
    
    print(f"\n   📏 Метрики (chars only - без спецсимволов и пробелов):")
    print(f"      Accuracy: {acc_chars_only*100:.2f}%")
    print(f"      CER: {avg_cer_chars_only:.4f}")
    print(f"      WER: {avg_wer_chars_only:.4f}")
    if avg_cer > 0:
        print(f"      Улучшение CER: {(avg_cer - avg_cer_chars_only)/avg_cer*100:.1f}%")
    if avg_wer > 0:
        print(f"      Улучшение WER: {(avg_wer - avg_wer_chars_only)/avg_wer*100:.1f}%")
    
    # 2. Типы ошибок
    print(f"\n2️⃣ ТИПЫ ОШИБОК:")
    error_types = analyze_error_types(error_details)
    print(f"   Разная длина: {error_types['length_mismatch']} ({error_types['length_mismatch']/error_types['total']*100:.1f}%)")
    print(f"   Только регистр: {error_types['case_errors']} ({error_types['case_errors']/error_types['total']*100:.1f}%)")
    print(f"   Похожие (1-2 символа): {error_types['similar_chars']} ({error_types['similar_chars']/error_types['total']*100:.1f}%)")
    print(f"   Полностью неверные: {error_types['completely_wrong']} ({error_types['completely_wrong']/error_types['total']*100:.1f}%)")
    
    # 3. Анализ длины слов с ошибками
    print(f"\n3️⃣ ДЛИНА СЛОВ С ОШИБКАМИ:")
    error_lengths = analyze_word_lengths(error_details)
    if error_lengths:
        avg_error_len = sum(error_lengths) / len(error_lengths)
        print(f"   Средняя длина: {avg_error_len:.1f} символов")
        print(f"   Мин: {min(error_lengths)}, Макс: {max(error_lengths)}")
        
        # Распределение по длинам
        length_dist = Counter(error_lengths)
        print(f"   Распределение:")
        for length in sorted(length_dist.keys())[:10]:  # Топ-10
            print(f"      {length} символов: {length_dist[length]} слов")
    
    # 4. Анализ ошибок по символам
    print(f"\n4️⃣ АНАЛИЗ ОШИБОК ПО СИМВОЛАМ:")
    substitutions, insertions, deletions, error_positions = analyze_character_errors(refs, hyps)
    
    # Позиции ошибок
    total_pos = sum(error_positions.values())
    if total_pos > 0:
        print(f"   Позиция ошибок в слове:")
        print(f"      Начало (0-20%): {error_positions['start']} ({error_positions['start']/total_pos*100:.1f}%)")
        print(f"      Середина (20-80%): {error_positions['middle']} ({error_positions['middle']/total_pos*100:.1f}%)")
        print(f"      Конец (80-100%): {error_positions['end']} ({error_positions['end']/total_pos*100:.1f}%)")
    
    # Самые частые замены
    print(f"\n   🔄 Топ-20 замен символов (правильный → ошибочный):")
    
    # Разделим замены на регистровые и не регистровые
    case_substitutions = []
    non_case_substitutions = []
    
    for (correct, wrong), count in substitutions.items():
        if correct.lower() == wrong.lower() and correct != wrong:
            case_substitutions.append(((correct, wrong), count))
        else:
            non_case_substitutions.append(((correct, wrong), count))
    
    case_substitutions.sort(key=lambda x: x[1], reverse=True)
    non_case_substitutions.sort(key=lambda x: x[1], reverse=True)
    
    if case_substitutions:
        print(f"\n      Регистровые замены (топ-10):")
        for (correct, wrong), count in case_substitutions[:10]:
            print(f"         '{correct}' → '{wrong}': {count} раз")
    
    if non_case_substitutions:
        print(f"\n      Другие замены (топ-20):")
        for (correct, wrong), count in non_case_substitutions[:20]:
            print(f"         '{correct}' → '{wrong}': {count} раз")
    
    # Самые частые вставки
    if insertions:
        print(f"\n   ➕ Топ-10 лишних символов:")
        for char, count in insertions.most_common(10):
            print(f"      '{char}': {count} раз")
    
    # Самые частые удаления
    if deletions:
        print(f"\n   ➖ Топ-10 пропущенных символов:")
        for char, count in deletions.most_common(10):
            print(f"      '{char}': {count} раз")
    
    # 5. Худшие примеры
    print(f"\n5️⃣ ХУДШИЕ ПРИМЕРЫ (топ-10 по CER):")
    worst_examples = sorted(error_details, key=lambda x: x['cer'], reverse=True)[:10]
    for i, ex in enumerate(worst_examples, 1):
        print(f"   {i}. [{ex['fname']}]")
        print(f"      GT:   '{ex['ref']}'")
        print(f"      Pred: '{ex['hyp']}'")
        print(f"      CER: {ex['cer']:.3f}, Conf: {ex['confidence']:.3f}")

    # === 5. ВСЕ ОШИБКИ (разбитые на 4 HTML, отсортированы по GT) ===
    print(f"\n5️⃣ СОЗДАЁМ HTML-ОТЧЁТЫ СО ВСЕМИ ОШИБКАМИ (разбитые на 4 части, сортировка по GT)...")

    import base64
    from io import BytesIO
    from PIL import Image
    import math

    # === 1. Берём все ошибки, сортируем по GT ===
    all_errors = sorted(error_details, key=lambda x: x['ref'].lower())
    num_errors = len(all_errors)
    num_parts = 4
    part_size = math.ceil(num_errors / num_parts)

    print(f"   Всего ошибок: {num_errors}")
    print(f"   Будет создано {num_parts} HTML-файла по ~{part_size} записей каждый")

    # === 2. Общий стиль и JS (единые для всех частей) ===
    def make_html_header(title):
        return [
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; background: #fafafa; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; table-layout: fixed; }",
            "th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: middle; overflow-wrap: break-word; }",
            "th { background-color: #f2f2f2; }",
            "td:nth-child(2) { width: 150px; text-align: center; }",
            "img { max-width: 140px; max-height: 80px; object-fit: contain; border-radius: 6px; background: #fff; }",
            ".gt { color: #006400; font-weight: bold; }",
            ".pred { color: #8B0000; font-weight: bold; }",
            ".edit { background: #ffffe0; }",
            ".num { text-align: center; }",
            "button { margin: 10px; padding: 6px 10px; }",
            "</style></head><body>",
            f"<h2>{title}</h2>",
            "<div>",
            "<button onclick='resizeImages(0.5)'>🔍 Уменьшить</button>",
            "<button onclick='resizeImages(1)'>🔎 Нормально</button>",
            "<button onclick='resizeImages(2)'>🔍 Увеличить</button>",
            "<button onclick='downloadCorrections()'>💾 Скачать правки (CSV)</button>",
            "</div>",
            "<script>",
            "function resizeImages(scale){document.querySelectorAll('img').forEach(img=>{img.style.maxWidth=(140*scale)+'px';img.style.maxHeight=(80*scale)+'px';});}",
            "function saveCorrection(id){const val=document.getElementById('edit_'+id).innerText.trim();localStorage.setItem('ocr_edit_'+id,val);}",
            "function loadCorrections(){document.querySelectorAll('[id^=edit_]').forEach(el=>{const saved=localStorage.getItem('ocr_edit_'+el.id.split('edit_')[1]);if(saved){el.innerText=saved;}});}",
            "function downloadCorrections(){let rows=[['#','filename','GT','Pred','CER','Conf','Correction']];document.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.getAttribute('data-id');const cells=tr.querySelectorAll('td');const correction=document.getElementById('edit_'+id).innerText.trim().replace(/\\n/g,' ');rows.push([id,cells[2].innerText,cells[3].innerText,cells[4].innerText,cells[5].innerText,cells[6].innerText,correction]);});const csvContent=rows.map(r=>r.map(v=>'\"'+v.replaceAll('\"','\"\"')+'\"').join(',')).join('\\n');const blob=new Blob([csvContent],{type:'text/csv;charset=utf-8;'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ocr_corrections.csv';a.click();}",
            "window.onload=loadCorrections;",
            "</script>",
            "<table>",
            "<tr><th>#</th><th>Изображение</th><th>Файл</th><th>GT</th><th>Pred</th><th>CER</th><th>Conf.</th><th>Правка ✏️</th></tr>"
        ]

    def make_html_row(i, ex):
        fname = ex['fname']
        cer = f"{ex['cer']:.3f}"
        conf = f"{ex['confidence']:.3f}"
        gt = ex['ref'].replace("<", "&lt;").replace(">", "&gt;")
        pred = ex['hyp'].replace("<", "&lt;").replace(">", "&gt;")

        # Находим путь к изображению
        img_path = None
        for d in datasets:
            candidate = os.path.join(d["image_dir"], fname)
            if os.path.exists(candidate):
                img_path = candidate
                break

        # Конвертируем в base64
        if img_path:
            try:
                with Image.open(img_path) as img:
                    img.thumbnail((400, 200))
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=80)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    img_tag = f"<img src='data:image/jpeg;base64,{img_base64}'>"
            except Exception:
                img_tag = f"<div style='color:red;'>Ошибка загрузки</div>"
        else:
            img_tag = "<div style='color:gray;'>Нет изображения</div>"

        return (
            f"<tr data-id='{i}'>"
            f"<td class='num'>{i}</td>"
            f"<td>{img_tag}</td>"
            f"<td>{fname}</td>"
            f"<td class='gt'>{gt}</td>"
            f"<td class='pred'>{pred}</td>"
            f"<td class='num'>{cer}</td>"
            f"<td class='num'>{conf}</td>"
            f"<td class='edit' id='edit_{i}' contenteditable='true' oninput='saveCorrection({i})'></td>"
            f"</tr>"
        )

    # === 3. Генерация 4 HTML-файлов ===
    for part_idx in range(num_parts):
        start = part_idx * part_size
        end = min(start + part_size, num_errors)
        subset = all_errors[start:end]

        if not subset:
            continue

        html_lines = make_html_header(f"📊 OCR ошибки (часть {part_idx+1} из {num_parts}) — записи {start+1}–{end}")
        for i, ex in enumerate(subset, start + 1):
            html_lines.append(make_html_row(i, ex))
        html_lines.append("</table></body></html>")

        html_path = os.path.join(
            os.path.dirname(model_path),
            f"ocr_all_errors_part{part_idx+1}.html"
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        print(f"💾 HTML-отчёт сохранён: {html_path}")

    print(f"\n✅ Создано {num_parts} HTML-файлов со всеми {num_errors} ошибками (отсортировано по GT).")


    # 6. Связь уверенности и ошибок
    print(f"\n6️⃣ СВЯЗЬ УВЕРЕННОСТИ И ОШИБОК:")
    low_conf_errors = [e for e in error_details if e['confidence'] < 0.8]
    high_conf_errors = [e for e in error_details if e['confidence'] >= 0.8]
    print(f"   Ошибки с низкой уверенностью (<0.8): {len(low_conf_errors)} ({len(low_conf_errors)/len(error_details)*100:.1f}%)")
    print(f"   Ошибки с высокой уверенностью (≥0.8): {len(high_conf_errors)} ({len(high_conf_errors)/len(error_details)*100:.1f}%)")
    
    if error_details:
        avg_conf_errors = sum(e['confidence'] for e in error_details) / len(error_details)
        print(f"   Средняя уверенность на ошибках: {avg_conf_errors:.3f}")
   
    # 7. Все ошибки, отсортированные по уверенности
    print(f"\n7️⃣ ВСЕ ОШИБКИ (отсортированные по уверенности модели):")
    sorted_errors = sorted(error_details, key=lambda x: x['confidence'], reverse=True)
    
    print(f"{'Файл':30s} | {'Conf.':>7s} | {'CER':>5s} | {'GT':25s} | {'Pred':25s}")
    print("-" * 100)
    # === Сохранение ошибок в CSV ===
    import csv

    output_csv = os.path.join(os.path.dirname(model_path), "ocr_errors_by_confidence.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "confidence", "CER", "WER", "GT", "Prediction"])
        for err in sorted_errors:
            writer.writerow([
                err['fname'],
                f"{err['confidence']:.4f}",
                f"{err['cer']:.4f}",
                f"{err['wer']:.4f}",
                err['ref'],
                err['hyp'],
            ])

    print(f"\n💾 Ошибки сохранены в файл: {output_csv}")
else:
    print("\n✅ Нет ошибок! Все слова распознаны идеально!")

print("\n" + "="*80)