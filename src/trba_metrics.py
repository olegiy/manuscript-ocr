import os
import time
import csv
import re
from collections import Counter, defaultdict
from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)
from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from manuscript.recognizers._trba.data.transforms import load_charset
import Levenshtein
from tqdm import tqdm


def normalize_text_letters_only(text: str) -> str:
    """
    Нормализует текст: оставляет только буквы, приводит к нижнему регистру.
    Удаляет пробелы, пунктуацию и цифры.
    """
    # Оставляем только буквы (латиница + кириллица + др. Unicode буквы)
    letters_only = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\u0080-\uFFFF]', '', text)
    return letters_only.lower()


# === Пути ===
# Можно указать несколько датасетов - они будут объединены
datasets = [
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\labels.csv",
    },
    {
        "image_dir": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img",
        "gt_path": r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv",
    },
    {
        "image_dir": r"C:\shared\orig_cyrillic\test",
        "gt_path": r"C:\shared\orig_cyrillic\test.tsv",  # Тот же файл что при обучении
    },
    {
        "image_dir": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val",
        "gt_path": r"C:\shared\school_notebooks_RU\school_notebooks_RU\val_converted.csv",
    },
]

model_path = r"C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx"
config_path = r"C:\Users\USER\Desktop\trba_exp_lite\config.json"
charset_path = r"C:\Users\USER\Desktop\trba_exp_lite\charset.txt"

batch_size = 64

# === Загружаем charset для создания датасетов ===
print("📚 Загрузка charset...")
itos, stoi = load_charset(charset_path)
print(f"   Размер алфавита: {len(itos)} символов")

# === Читаем данные используя тот же OCRDatasetAttn что и при обучении ===
all_samples = []  # Список кортежей (image_path, label, dataset_idx)
image_to_dataset = {}  # Маппинг: путь к изображению → индекс датасета

for idx, dataset in enumerate(datasets, 1):
    image_dir = dataset["image_dir"]
    gt_path = dataset["gt_path"]
    
    print(f"\n📂 Датасет {idx}: {os.path.basename(image_dir)}")
    print(f"   CSV: {os.path.basename(gt_path)}")
    
    # Создаём датасет с ТОЧНО ТЕМИ ЖЕ настройками что при обучении!
    # ВАЖНО: strict_max_len=True и max_len=40 как при обучении
    ds = OCRDatasetAttn(
        csv_path=gt_path,
        images_dir=image_dir,
        stoi=stoi,
        img_height=32,
        img_max_width=256,  # Как в config.json
        transform=None,
        has_header=None,  # Автоопределение
        encoding="utf-8",
        delimiter=None,  # Автоопределение по расширению
        strict_charset=False,  # Не фильтруем по алфавиту
        validate_image=False,  # Не проверяем изображения
        max_len=40,  # КАК ПРИ ОБУЧЕНИИ!
        strict_max_len=True,  # КАК ПРИ ОБУЧЕНИИ! Фильтруем длинные тексты
        num_workers=0
    )
    
    print(f"   ✅ Загружено {len(ds)} валидных примеров")
    
    # Показываем статистику фильтрации (если были отбросы)
    if hasattr(ds, '_reasons'):
        total_filtered = sum(ds._reasons.values())
        if total_filtered > 0:
            print(f"   ⚠️  Отфильтровано {total_filtered} примеров:")
            for reason, count in ds._reasons.items():
                if count > 0:
                    print(f"      - {reason}: {count}")
    
    # Сохраняем пути к изображениям и метки
    for i in range(len(ds)):
        img_path, label = ds.samples[i]
        all_samples.append((img_path, label, idx))
        image_to_dataset[img_path] = idx

# === Формируем списки для распознавания ===
images = [sample[0] for sample in all_samples]
gt_data = {os.path.basename(sample[0]): sample[1] for sample in all_samples}

# === Ограничиваем количество изображений ===
max_images = 1000000000000
if len(images) > max_images:
    print(f"\n⚠️ Берём только первые {max_images} изображений из {len(images)}")
    images = images[:max_images]
    all_samples = all_samples[:max_images]

if not images:
    raise RuntimeError(f"❌ Не найдено изображений ни в одном датасете!")

print(f"\n📁 ИТОГО: {len(images)} изображений для распознавания")

# === Инициализация модели ===
print(f"\n🔧 Инициализация модели...")
recognizer = TRBA(weights_path=model_path, config_path=config_path, charset_path=charset_path)

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

# === Сопоставляем с ground truth ===
refs, hyps = [], []
dataset_ids = []  # Индекс датасета для каждого примера
total_cer, total_wer = 0.0, 0.0
cer_count, wer_count = 0, 0
error_details = []  # Для детального анализа ошибок

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
    dataset_ids.append(image_to_dataset.get(path, 0))  # Запоминаем датасет

    cer = character_error_rate(ref_text, pred_text)
    wer = word_error_rate(ref_text, pred_text)

    total_cer += cer
    total_wer += wer
    cer_count += 1
    wer_count += 1
    
    # Сохраняем детали для анализа
    if ref_text != pred_text:
        error_details.append({
            'fname': fname,
            'ref': ref_text,
            'hyp': pred_text,
            'cer': cer,
            'wer': wer,
            'confidence': score,
            'dataset_id': image_to_dataset.get(path, 0)
        })

    print(
        f"{fname:40s} → {pred_text:20s} | GT: {ref_text:20s} | CER={cer:.3f} | WER={wer:.3f}"
    )

# === Метрики ===
acc = compute_accuracy(refs, hyps)

# Регистронезависимая точность (case-insensitive)
acc_case_insensitive = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower()) / max(len(refs), 1)

# Точность только по буквам (без пунктуации, пробелов, регистронезависимо)
acc_letters_only = sum(
    1 for r, h in zip(refs, hyps) 
    if normalize_text_letters_only(r) == normalize_text_letters_only(h)
) / max(len(refs), 1)

# Точность с учетом только регистра (когда все символы верны, но регистр другой)
case_only_errors = sum(1 for r, h in zip(refs, hyps) if r.lower() == h.lower() and r != h)

avg_cer = total_cer / max(cer_count, 1)
avg_wer = total_wer / max(wer_count, 1)

# === Вычисление метрик по датасетам ===
def compute_dataset_metrics(refs, hyps, dataset_ids, dataset_idx):
    """Вычисляет метрики для конкретного датасета"""
    # Фильтруем только примеры из этого датасета
    dataset_refs = [r for r, h, d in zip(refs, hyps, dataset_ids) if d == dataset_idx]
    dataset_hyps = [h for r, h, d in zip(refs, hyps, dataset_ids) if d == dataset_idx]
    
    if not dataset_refs:
        return None
    
    # Базовые метрики
    acc = compute_accuracy(dataset_refs, dataset_hyps)
    acc_ci = sum(1 for r, h in zip(dataset_refs, dataset_hyps) if r.lower() == h.lower()) / len(dataset_refs)
    acc_letters = sum(
        1 for r, h in zip(dataset_refs, dataset_hyps) 
        if normalize_text_letters_only(r) == normalize_text_letters_only(h)
    ) / len(dataset_refs)
    
    # CER и WER
    total_cer = sum(character_error_rate(r, h) for r, h in zip(dataset_refs, dataset_hyps))
    total_wer = sum(word_error_rate(r, h) for r, h in zip(dataset_refs, dataset_hyps))
    avg_cer = total_cer / len(dataset_refs)
    avg_wer = total_wer / len(dataset_refs)
    
    # CER case-insensitive
    total_cer_ci = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(dataset_refs, dataset_hyps))
    avg_cer_ci = total_cer_ci / len(dataset_refs)
    
    # CER letters only
    total_cer_letters = 0.0
    for r, h in zip(dataset_refs, dataset_hyps):
        r_letters = normalize_text_letters_only(r)
        h_letters = normalize_text_letters_only(h)
        if r_letters:
            total_cer_letters += character_error_rate(r_letters, h_letters)
    avg_cer_letters = total_cer_letters / len(dataset_refs)
    
    return {
        'count': len(dataset_refs),
        'acc': acc,
        'acc_ci': acc_ci,
        'acc_letters': acc_letters,
        'cer': avg_cer,
        'cer_ci': avg_cer_ci,
        'cer_letters': avg_cer_letters,
        'wer': avg_wer
    }

print("\n" + "="*120)
print("📊 МЕТРИКИ ПО ДАТАСЕТАМ")
print("="*120)

# Вычисляем метрики для каждого датасета
dataset_metrics = {}
for idx in range(1, len(datasets) + 1):
    metrics = compute_dataset_metrics(refs, hyps, dataset_ids, idx)
    if metrics:
        dataset_metrics[idx] = metrics

# Общие метрики
overall_metrics = {
    'count': len(refs),
    'acc': acc,
    'acc_ci': acc_case_insensitive,
    'acc_letters': acc_letters_only,
    'cer': avg_cer,
    'wer': avg_wer
}

# Вычисляем общие CER case-insensitive и letters only для таблицы
total_cer_ci = sum(character_error_rate(r.lower(), h.lower()) for r, h in zip(refs, hyps))
overall_metrics['cer_ci'] = total_cer_ci / max(len(refs), 1)

total_cer_letters = 0.0
for r, h in zip(refs, hyps):
    r_letters = normalize_text_letters_only(r)
    h_letters = normalize_text_letters_only(h)
    if r_letters:
        total_cer_letters += character_error_rate(r_letters, h_letters)
overall_metrics['cer_letters'] = total_cer_letters / max(len(refs), 1)

# Печатаем таблицу
print(f"\n{'Датасет':<30} {'Count':>8} {'Acc':>8} {'Acc-CI':>8} {'Acc-L':>8} {'CER':>8} {'CER-CI':>8} {'CER-L':>8} {'WER':>8}")
print("-" * 120)

for idx in sorted(dataset_metrics.keys()):
    dataset = datasets[idx - 1]
    dataset_name = os.path.basename(dataset["image_dir"])[:28]
    m = dataset_metrics[idx]
    print(f"{dataset_name:<30} {m['count']:>8} {m['acc']*100:>7.2f}% {m['acc_ci']*100:>7.2f}% {m['acc_letters']*100:>7.2f}% {m['cer']:>8.4f} {m['cer_ci']:>8.4f} {m['cer_letters']:>8.4f} {m['wer']:>8.4f}")

print("-" * 120)
m = overall_metrics
print(f"{'OVERALL':<30} {m['count']:>8} {m['acc']*100:>7.2f}% {m['acc_ci']*100:>7.2f}% {m['acc_letters']*100:>7.2f}% {m['cer']:>8.4f} {m['cer_ci']:>8.4f} {m['cer_letters']:>8.4f} {m['wer']:>8.4f}")
print("="*120)

print("\nЛегенда:")
print("  Acc      - Accuracy (case-sensitive)")
print("  Acc-CI   - Accuracy (case-insensitive)")
print("  Acc-L    - Accuracy (letters only, case-insensitive)")
print("  CER      - Character Error Rate")
print("  CER-CI   - CER (case-insensitive)")
print("  CER-L    - CER (letters only)")
print("  WER      - Word Error Rate")

print("\n=== Дополнительная информация ===")
print(f"Case-only errors:                        {case_only_errors} ({case_only_errors/max(len(refs), 1)*100:.2f}%)")
print(f"Processed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")

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
    
    # Статистика ошибок по датасетам
    print(f"\n   По датасетам:")
    for idx in sorted(dataset_metrics.keys()):
        dataset = datasets[idx - 1]
        dataset_name = os.path.basename(dataset["image_dir"])
        dataset_errors = [e for e in error_details if e['dataset_id'] == idx]
        dataset_total = dataset_metrics[idx]['count']
        print(f"      {dataset_name}: {len(dataset_errors)}/{dataset_total} ошибок ({len(dataset_errors)/max(dataset_total, 1)*100:.1f}%)")
    
    # Регистронезависимые метрики
    total_cer_ci = 0.0
    total_wer_ci = 0.0
    total_cer_letters = 0.0
    for ref, hyp in zip(refs, hyps):
        total_cer_ci += character_error_rate(ref.lower(), hyp.lower())
        total_wer_ci += word_error_rate(ref.lower(), hyp.lower())
        
        # Метрики только по буквам
        ref_letters = normalize_text_letters_only(ref)
        hyp_letters = normalize_text_letters_only(hyp)
        if ref_letters:  # Избегаем деления на ноль для пустых строк
            total_cer_letters += character_error_rate(ref_letters, hyp_letters)
    
    avg_cer_ci = total_cer_ci / max(len(refs), 1)
    avg_wer_ci = total_wer_ci / max(len(refs), 1)
    avg_cer_letters = total_cer_letters / max(len(refs), 1)
    
    print(f"\n   📏 Метрики (case-sensitive):")
    print(f"      Accuracy: {acc*100:.2f}%")
    print(f"      CER: {avg_cer:.4f}")
    print(f"      WER: {avg_wer:.4f}")
    
    print(f"\n   📏 Метрики (case-insensitive):")
    print(f"      Accuracy: {acc_case_insensitive*100:.2f}%")
    print(f"      CER: {avg_cer_ci:.4f}")
    print(f"      WER: {avg_wer_ci:.4f}")
    print(f"      Улучшение CER: {(avg_cer - avg_cer_ci)/avg_cer*100:.1f}%")
    print(f"      Улучшение WER: {(avg_wer - avg_wer_ci)/avg_wer*100:.1f}%")
    
    print(f"\n   📏 Метрики (letters only, case-insensitive):")
    print(f"      Accuracy: {acc_letters_only*100:.2f}%")
    print(f"      CER: {avg_cer_letters:.4f}")
    print(f"      Улучшение CER: {(avg_cer - avg_cer_letters)/avg_cer*100:.1f}%")
    
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