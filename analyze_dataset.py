"""
analyze_dataset.py - Анализ сгенерированного датасета.
Показывает распределение решений, тиров, позиций.
Запуск: python analyze_dataset.py
"""
import json
import re
import os
from collections import Counter

DATASET_FILE = "dataset_preflop.json"


def load_dataset(filename):
    """Загружает датасет с диска."""
    if not os.path.exists(filename):
        print(f"❌ Файл {filename} не найден!")
        print("💡 Сначала запустите: python dataset_preflop.py")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_action(completion):
    """Извлекает действие из completion (ответа)."""
    match = re.search(r'ДЕЙСТВИЕ:\s*(\w+)', completion)
    return match.group(1) if match else "UNKNOWN"


def extract_amount(completion):
    """Извлекает сумму из completion."""
    match = re.search(r'СУММА:\s*(\d+)', completion)
    return int(match.group(1)) if match else 0


def analyze_dataset(examples):
    """Полный анализ датасета."""
    total = len(examples)
    print("=" * 60)
    print(f"📊 АНАЛИЗ ДАТАСЕТА: {total} примеров")
    print("=" * 60)

    # === 1. Распределение по действиям ===
    actions = Counter()
    amounts_by_action = {}
    for ex in examples:
        action = extract_action(ex["completion"])
        actions[action] += 1
        amount = extract_amount(ex["completion"])
        if action not in amounts_by_action:
            amounts_by_action[action] = []
        amounts_by_action[action].append(amount)

    print("\n🎯 1. РАСПРЕДЕЛЕНИЕ ПО ДЕЙСТВИЯМ:")
    print("-" * 40)
    for action, count in actions.most_common():
        percent = count / total * 100
        amounts = amounts_by_action[action]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        print(f"  {action:<8} {count:>5} примеров  ({percent:>5.1f}%)   средняя сумма: ${avg_amount:.0f}")

    # === 2. Распределение по тирам ===
    tiers = Counter()
    tier_actions = {}
    for ex in examples:
        tier = ex["metadata"]["tier"]
        tiers[tier] += 1
        action = extract_action(ex["completion"])
        if tier not in tier_actions:
            tier_actions[tier] = Counter()
        tier_actions[tier][action] += 1

    print("\n🃏 2. РАСПРЕДЕЛЕНИЕ ПО ТИРАМ РУК:")
    print("-" * 40)
    for tier in sorted(tiers.keys()):
        count = tiers[tier]
        percent = count / total * 100
        print(f"  Тир {tier}: {count:>5} примеров ({percent:>5.1f}%)")

    # === 3. Матрица: Тир × Действие ===
    print("\n📋 3. КАК РЕШАЕТ «УЧИТЕЛЬ» ПО ТИРАМ:")
    print("-" * 40)
    print(f"  {'Тир':<6}{'FOLD':>8}{'CALL':>8}{'RAISE':>8}{'CHECK':>8}")
    all_actions = ["FOLD", "CALL", "RAISE", "CHECK"]
    for tier in sorted(tier_actions.keys()):
        row = f"  Тир {tier:<2}"
        for action in all_actions:
            count = tier_actions[tier].get(action, 0)
            tier_total = tiers[tier]
            percent = count / tier_total * 100 if tier_total > 0 else 0
            row += f"{percent:>7.0f}%"
        print(row)

    # === 4. Распределение по позициям ===
    positions = Counter()
    pos_actions = {}
    for ex in examples:
        pos = ex["metadata"]["position"]
        positions[pos] += 1
        action = extract_action(ex["completion"])
        if pos not in pos_actions:
            pos_actions[pos] = Counter()
        pos_actions[pos][action] += 1

    print("\n📍 4. РАСПРЕДЕЛЕНИЕ ПО ПОЗИЦИЯМ:")
    print("-" * 40)
    for pos in ["UTG", "MP", "CO", "BTN", "SB", "BB"]:
        if pos in positions:
            count = positions[pos]
            actions_str = ", ".join(
                f"{a}: {pos_actions[pos].get(a, 0)}" for a in all_actions
            )
            print(f"  {pos:<4} {count:>4} примеров | {actions_str}")

    # === 5. Примеры каждого действия ===
    print("\n📝 5. ПРИМЕРЫ КАЖДОГО ДЕЙСТВИЯ:")
    print("-" * 40)
    shown_actions = set()
    for ex in examples:
        action = extract_action(ex["completion"])
        if action not in shown_actions:
            shown_actions.add(action)
            print(f"\n  [{action}] Рука: {ex['metadata']['hand']}")
            print(f"  Позиция: {ex['metadata']['position']}")
            print(f"  Ситуация: {ex['metadata']['action_before']}")
            print(f"  Ответ учителя:")
            for line in ex["completion"].split("\n"):
                print(f"    {line}")
        if len(shown_actions) >= 3:
            break

    # === 6. Проверка качества ===
    print("\n✅ 6. ПРОВЕРКА КАЧЕСТВА ДАТАСЕТА:")
    print("-" * 40)

    issues = []

    # Проверка: премиум-руки не должны фолдить
    premium_folds = 0
    for ex in examples:
        if ex["metadata"]["tier"] == 1 and extract_action(ex["completion"]) == "FOLD":
            premium_folds += 1
    if premium_folds > 0:
        issues.append(f"❌ Премиум-руки фолдятся: {premium_folds} случаев")
    else:
        print("  ✓ Премиум-руки (тир 1) никогда не фолдятся")

    # Проверка: мусор не должен рейзить
    trash_raises = 0
    for ex in examples:
        if ex["metadata"]["tier"] == 7 and extract_action(ex["completion"]) == "RAISE":
            trash_raises += 1
    if trash_raises > 0:
        issues.append(f"❌ Мусорные руки рейзятся: {trash_raises} случаев")
    else:
        print("  ✓ Мусорные руки (тир 7) никогда не рейзятся")

    # Проверка: баланс датасета
    fold_percent = actions.get("FOLD", 0) / total * 100
    if fold_percent > 70:
        issues.append(f"⚠️ Датасет перекошен в фолды ({fold_percent:.0f}%)")
    else:
        print(f"  ✓ Баланс нормальный (фолдов {fold_percent:.0f}%)")

    if issues:
        print("\n  ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n  🎉 Датасет качественный, можно обучать!")

    # === Итоговый вывод для копирования ===
    print("\n" + "=" * 60)
    print("📋 ИТОГ (скопируйте это и отправьте мне):")
    print("=" * 60)
    print(f"Всего примеров: {total}")
    print(f"Действия: {dict(actions)}")
    print(f"Тиры: {dict(sorted(tiers.items()))}")
    print(f"Фолды: {fold_percent:.1f}%")
    print(f"Коллы: {actions.get('CALL', 0) / total * 100:.1f}%")
    print(f"Рейзы: {actions.get('RAISE', 0) / total * 100:.1f}%")
    if issues:
        print(f"Проблемы: {issues}")
    else:
        print("Проблемы: нет")


if __name__ == "__main__":
    examples = load_dataset(DATASET_FILE)
    if examples:
        analyze_dataset(examples)