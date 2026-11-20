"""
check_distribution.py - Проверка распределения действий в датасете.
Запуск: python train_2/check_distribution.py
"""
import sys
import os

# Добавляем train_2 в path (чтобы видеть deep_analyzer)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_dataset_v2 import generate_unified_dataset
from collections import Counter

# Генерируем 500 примеров (быстро, ~1 минута)
print("Генерация 500 примеров...")
examples = generate_unified_dataset(count_per_stage=500)

# Считаем действия
actions = Counter()
for x in examples:
    for line in x['completion'].split('\n'):
        if line.startswith('ДЕЙСТВИЕ:'):
            action = line.split(':')[1].strip()
            actions[action] += 1

total = sum(actions.values())
print(f"\n📊 Распределение действий ({total} примеров):")
for action, count in actions.most_common():
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f"  {action:8} {count:5} ({pct:5.1f}%) {bar}")

# Проверка
print("\n✅ Проверка:")
if actions.get('FOLD', 0) / total < 0.15:
    print(f"  ❌ FOLD = {actions.get('FOLD', 0)/total*100:.1f}% — мало (должно быть >15%)")
else:
    print(f"  ✅ FOLD = {actions.get('FOLD', 0)/total*100:.1f}%")

if actions.get('CALL', 0) / total < 0.10:
    print(f"  ❌ CALL = {actions.get('CALL', 0)/total*100:.1f}% — мало (должно быть >10%)")
else:
    print(f"  ✅ CALL = {actions.get('CALL', 0)/total*100:.1f}%")