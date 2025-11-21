# Быстрый бенчмарк TRBA - измеряет скорость и 9 метрик точности
#
# ПРИМЕРЫ ЗАПУСКА ДЛЯ ДАТАСЕТОВ:
#
# 1. orig_cyrillic test:
#    python scripts/trba_infer_speed_test.py --folder "C:\shared\orig_cyrillic\test" --gt-csv "C:\shared\orig_cyrillic\test.csv" --batch-size 64
#
# 2. school_notebooks_RU val:
#    python scripts/trba_infer_speed_test.py --folder "C:\shared\school_notebooks_RU\school_notebooks_RU\val" --gt-csv "C:\shared\school_notebooks_RU\school_notebooks_RU\val_converted.csv" --batch-size 64
#
# 3. archive printed val:
#    python scripts/trba_infer_speed_test.py --folder "C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\img" --gt-csv "C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\labels.csv" --batch-size 64
#
# 4. archive handwritten val:
#    python scripts/trba_infer_speed_test.py --folder "C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img" --gt-csv "C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv" --batch-size 64
#
# 5. С кастомной моделью (ONNX):
#    python scripts/trba_infer_speed_test.py --folder "C:\dataset\val\img" --gt-csv "C:\dataset\val\labels.csv" --model-path "C:\Users\USER\Desktop\trba_exp_lite\trba_exp_lite.onnx" --config-path "C:\Users\USER\Desktop\trba_exp_lite\config.json" --batch-size 64
#
# 6. С кастомной моделью (PTH):
#    python scripts/trba_infer_speed_test.py --folder "C:\dataset\val\img" --gt-csv "C:\dataset\val\labels.csv" --model-path "C:\Users\USER\Desktop\trba_exp_lite\best_acc_weights.pth" --config-path "C:\Users\USER\Desktop\trba_exp_lite\config.json" --batch-size 64
#
# 7. Только скорость (без GT):
#    python scripts/trba_infer_speed_test.py --folder "C:\dataset\val\img" --batch-size 64

import argparse, sys, time, csv, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import character_error_rate, word_error_rate, compute_accuracy

def filter_chars_only(text):
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯѣѢіІѳѲѵѴѫѪѭѬѯѮѱѰѡѠѕѕѧѦѩѨ0123456789')
    return ''.join(c for c in text if c in allowed)

def load_gt(path):
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                gt[row[0].strip()] = ','.join(row[1:]).strip()
    return gt

def get_images(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}])

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
parser.add_argument('--gt-csv', default=None)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--model-path', default=None)
parser.add_argument('--config-path', default=None)
parser.add_argument('--cpu-only', action='store_true', help='Force CPU inference')
parser.add_argument('--gpu-only', action='store_true', help='Force GPU inference')
args = parser.parse_args()

print('='*80, '\nTRBA Benchmark\n', '='*80)
images = get_images(args.folder)
print(f'Found {len(images)} images')
gt_data = load_gt(args.gt_csv) if args.gt_csv else None
if gt_data: print(f'Loaded {len(gt_data)} GT labels')

kwargs = {}
if args.model_path: kwargs['model_path'] = args.model_path
if args.config_path: kwargs['config_path'] = args.config_path

# Determine device
if args.cpu_only:
    kwargs['device'] = 'cpu'
elif args.gpu_only:
    kwargs['device'] = 'cuda'

recognizer = TRBA(**kwargs)

print(f'Running inference (batch={args.batch_size})...')
t0 = time.perf_counter()
results = recognizer.predict(images, batch_size=args.batch_size)
t1 = time.perf_counter()
print(f'\nPERFORMANCE:\n  Total: {t1-t0:.3f}s\n  Per image: {(t1-t0)/len(images)*1000:.2f}ms\n  Throughput: {len(images)/(t1-t0):.2f} img/s')

if gt_data:
    refs, hyps = [], []
    for img, res in zip(images, results):
        fname = os.path.basename(img)
        if fname in gt_data:
            refs.append(gt_data[fname])
            hyps.append(res['text'])
    if refs:
        # Accuracy variants
        acc_cs = compute_accuracy(refs, hyps)
        acc_ci = sum(1 for r,h in zip(refs,hyps) if r.lower()==h.lower())/len(refs)
        rco = [filter_chars_only(r) for r in refs]
        hco = [filter_chars_only(h) for h in hyps]
        acc_co = sum(1 for r,h in zip(rco,hco) if r.lower()==h.lower())/len(refs)
        
        # WER variants
        wer_cs = sum(word_error_rate(r,h) for r,h in zip(refs,hyps))/len(refs)
        wer_ci = sum(word_error_rate(r.lower(),h.lower()) for r,h in zip(refs,hyps))/len(refs)
        wer_co = sum(word_error_rate(r,h) if r and h else (1.0 if r or h else 0.0) for r,h in zip(rco,hco))/len(refs)
        
        # NED variants (1 - CER)
        cer_cs = sum(character_error_rate(r,h) for r,h in zip(refs,hyps))/len(refs)
        cer_ci = sum(character_error_rate(r.lower(),h.lower()) for r,h in zip(refs,hyps))/len(refs)
        cer_co = sum(character_error_rate(r,h) if r and h else (1.0 if r or h else 0.0) for r,h in zip(rco,hco))/len(refs)
        ned_cs = 1.0 - cer_cs
        ned_ci = 1.0 - cer_ci
        ned_co = 1.0 - cer_co
        
        print(f'\nMETRICS (9):')
        print(f'{"":20} {"CS":>10} {"CI":>10} {"CO":>10}')
        print(f'{"-"*52}')
        print(f'{"Accuracy":<20} {acc_cs*100:>9.2f}% {acc_ci*100:>9.2f}% {acc_co*100:>9.2f}%')
        print(f'{"WER":<20} {wer_cs:>10.4f} {wer_ci:>10.4f} {wer_co:>10.4f}')
        print(f'{"NED (1-CER)":<20} {ned_cs*100:>9.2f}% {ned_ci*100:>9.2f}% {ned_co*100:>9.2f}%')
        print(f'{"-"*52}')
        print(f'Evaluated: {len(refs)} images')
        print(f'\nCS=case-sensitive, CI=case-insensitive, CO=chars-only')
print('\n[OK]')
