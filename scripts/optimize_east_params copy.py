import sys
from pathlib import Path

# Добавляем путь к scripts
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))

from optimize_east_params import run_optimization

'''

    '''
# Список датасетов: (путь_к_изображениям, путь_к_аннотациям)
DATASETS = [
    (
        r"C:\shared\data0205\data02065\Archives020525\train_images",
        r"C:\shared\data0205\data02065\Archives020525\train.json"
    ),
    (
        r"C:\shared\data0205\data02065\school_notebooks_RU\train_images",
        r"C:\shared\data0205\data02065\school_notebooks_RU\train.json"
    ),

    (
        r"C:\shared\data0205\data02065\TotalText\train_images",
        r"C:\shared\data0205\data02065\TotalText\train.json"
    ),
    (
        r"C:\shared\data0205\data02065\DDI_100\train_images",
        r"C:\shared\data0205\data02065\DDI_100\train.json"
    ),
    (
        r"C:\shared\data0205\data02065\IAM\train_images",
        r"C:\shared\data0205\data02065\IAM\train.json"
    ),
] 

N_IMAGES = 10        # Общее количество изображений
N_TRIALS = 500         # Количество итераций оптимизации
TARGET_SIZE = 1280    # Размер изображения для детектора
DEVICE = "cuda"        # "cpu" или "cuda"
OUTPUT = "optimization_results.json"
CACHE = "optuna_cache.json"  # Кеш (или None)
SEED = 42


results = run_optimization(
    datasets=DATASETS,
    n_images=N_IMAGES,
    n_trials=N_TRIALS,
    target_size=TARGET_SIZE,
    device=DEVICE,
    output=OUTPUT,
    cache=CACHE,
    seed=SEED,
)

print(f"Best F1@0.5:0.95 Score: {results['best_score']:.4f}")
print("\nBest Parameters:")
for param, value in results['best_params'].items():
    print(f"  {param}: {value}")


# Визуализация результатов (опционально, требует matplotlib)
import matplotlib.pyplot as plt

history = results['history']
scores = [h['f1_score'] for h in history]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(scores)
plt.xlabel('Trial')
plt.ylabel('F1@0.5:0.95')
plt.title('Optimization Progress')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(scores, bins=20)
plt.xlabel('F1@0.5:0.95')
plt.ylabel('Count')
plt.title('Score Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()