"""
test_unified.py - Проверка единого датасета.
"""
import json
from collections import Counter

with open("dataset_unified.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Всего примеров: {len(data)}")
print(f"По стадиям: {Counter(ex['stage'] for ex in data)}")

print("\n" + "=" * 60)
print("ПРИМЕР ПРЕФЛОП:")
print("=" * 60)
preflop = [ex for ex in data if ex["stage"] == "preflop"][0]
print(preflop["prompt"])
print("\nОТВЕТ:")
print(preflop["completion"])

print("\n" + "=" * 60)
print("ПРИМЕР ФЛОП:")
print("=" * 60)
flop = [ex for ex in data if ex["stage"] == "flop"][0]
print(flop["prompt"])
print("\nОТВЕТ:")
print(flop["completion"])

print("\n" + "=" * 60)
print("ПРИМЕР РИВЕР:")
print("=" * 60)
river = [ex for ex in data if ex["stage"] == "river"][0]
print(flop["prompt"])
print("\nОТВЕТ:")
print(flop["completion"])

print("\n" + "=" * 60)
print("ПРИМЕР ТЕРН:")
print("=" * 60)
tern = [ex for ex in data if ex["stage"] == "turn"][0]
print(flop["prompt"])
print("\nОТВЕТ:")
print(flop["completion"])

# Проверяем уникальность
unique_completions = len(set(ex["completion"] for ex in data))
print(f"\nУникальных completion: {unique_completions} ({unique_completions/len(data)*100:.1f}%)")