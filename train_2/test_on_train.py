# check_unified.py
import json
from collections import Counter

with open("dataset_unified.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Всего: {len(data)}")
print("По стадиям:", dict(Counter(ex["stage"] for ex in data)))
print(f"Уникальных completion: {len(set(ex['completion'] for ex in data))}")

# Проверка: все стадии представлены?
for stage in ["preflop", "flop", "turn", "river"]:
    stage_ex = [ex for ex in data if ex["stage"] == stage]
    if not stage_ex:
        print(f"❌ {stage} отсутствует!")
    else:
        print(f"✅ {stage}: {len(stage_ex)} примеров")
        # Проверяем что completion не пустой
        empty = sum(1 for ex in stage_ex if not ex["completion"].strip())
        if empty:
            print(f"   ⚠️  {empty} примеров с пустым completion")

# Проверяем что АНАЛИЗ разнообразный
analyses = [ex["completion"].split("\n")[0].replace("АНАЛИЗ: ", "") for ex in data]
unique_analyses = len(set(analyses))
print(f"\nУникальных АНАЛИЗ: {unique_analyses} из {len(analyses)} ({unique_analyses/len(analyses)*100:.1f}%)")