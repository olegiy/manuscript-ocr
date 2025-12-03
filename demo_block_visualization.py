"""
Демонстрация визуализации блоков в функции visualize_page.

Этот скрипт показывает разницу между визуализацией с show_order=True и show_order=False.
При show_order=True блоки отображаются с полупрозрачными прямоугольниками разных цветов.
"""
from manuscript.detectors import EAST
from manuscript.utils import visualize_page

# Путь к изображению
image_path = r"C:\Users\USER\Desktop\data02065\Archives020525\test_images\418.jpg"

# Запуск детектора
print("Запуск EAST детектора...")
result = EAST(weights="east_50_g1").predict(image_path)

# Информация о структуре
print(f"\nОбнаружено блоков: {len(result['page'].blocks)}")
for i, block in enumerate(result['page'].blocks):
    print(f"  Блок {i}: {len(block.lines)} строк")
    total_words = sum(len(line.words) for line in block.lines)
    print(f"           {total_words} слов")

# Визуализация БЕЗ порядка и блоков
print("\n1. Визуализация без порядка (show_order=False)...")
vis_no_order = visualize_page(
    image_path, 
    result["page"], 
    show_order=False,
)
vis_no_order.show()

# Визуализация С порядком и блоками
print("\n2. Визуализация с порядком и блоками (show_order=True)...")
vis_with_order = visualize_page(
    image_path, 
    result["page"], 
    show_order=True,
)
vis_with_order.show()

print("\nГотово!")
print("\nОсобенности визуализации при show_order=True:")
print("  - Каждая строка имеет свой цвет")
print("  - Слова пронумерованы в порядке чтения")
print("  - Слова соединены линиями")
print("  - Блоки обведены полупрозрачными прямоугольниками разных цветов")
print("  - Альфа блоков = 0.15 (15% непрозрачности)")
