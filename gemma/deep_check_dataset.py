"""
check_dataset.py - Полная проверка датасета:
1. Распределение стадий
2. Распределение действий
3. Уникальность
4. Длина в токенах (min/median/p90/p99/max)
5. Проверка MAX_SEQ_LEN (сколько примеров обрежется)
6. Оценка времени обучения
"""
import json
import sys
import os
from collections import Counter

# Добавляем путь к train_2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer

# ============================================================
# КОНФИГ
# ============================================================
DATASET_FILE = "./gemma/dataset_2000_v2.json"
TOKENIZER_NAME = "google/gemma-3-4b-it"
BATCH_SIZE = 1          # per_device_train_batch_size
GRAD_ACCUM = 8          # gradient_accumulation_steps
NUM_EPOCHS = 3
GPU_SPEED = 10.0        # сек/шаг (замерьте на вашей GPU)
MAX_SEQ_LEN = 788       # ← ваш MAX_SEQ_LEN

# prompt: max_length = 768 - 250 = 518 (prompt max 520 → влезает)
# completion: max_length = 250 (completion max 186 → влезает)

# ============================================================
# ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("  ПРОВЕРКА ДАТАСЕТА")
print("=" * 60)

with open(DATASET_FILE, encoding="utf-8") as f:
    data = json.load(f)

print(f"\n📂 Загружено: {len(data)} примеров")

# ============================================================
# 1. РАСПРЕДЕЛЕНИЕ ПО СТАДИЯМ
# ============================================================
print("\n" + "=" * 60)
print("1. РАСПРЕДЕЛЕНИЕ ПО СТАДИЯМ")
print("=" * 60)

stages = Counter(ex["stage"] for ex in data)
for stage, count in stages.most_common():
    pct = count / len(data) * 100
    bar = "█" * int(pct / 2)
    print(f"  {stage:10} {count:5} ({pct:5.1f}%) {bar}")

# ============================================================
# 2. РАСПРЕДЕЛЕНИЕ ПО ДЕЙСТВИЯМ
# ============================================================
print("\n" + "=" * 60)
print("2. РАСПРЕДЕЛЕНИЕ ПО ДЕЙСТВИЯМ")
print("=" * 60)

actions = Counter()
for ex in data:
    for line in ex["completion"].split("\n"):
        if line.strip().startswith("ДЕЙСТВИЕ:"):
            action = line.split(":", 1)[1].strip()
            actions[action] += 1
            break

total_actions = sum(actions.values())
for action, count in actions.most_common():
    pct = count / total_actions * 100
    bar = "█" * int(pct / 2)
    print(f"  {action:8} {count:5} ({pct:5.1f}%) {bar}")

# Проверка баланса
print("\n  ✅ Проверка баланса:")
checks = {
    "FOLD": (10, 40),
    "CALL": (10, 30),
    "RAISE": (10, 30),
    "BET": (10, 40),
    "CHECK": (10, 40),
}
for action, (low, high) in checks.items():
    pct = actions.get(action, 0) / total_actions * 100
    if pct < low:
        print(f"    ❌ {action}: {pct:.1f}% < {low}% (мало)")
    elif pct > high:
        print(f"    ⚠️ {action}: {pct:.1f}% > {high}% (много)")
    else:
        print(f"    ✅ {action}: {pct:.1f}%")

# ============================================================
# 3. УНИКАЛЬНОСТЬ
# ============================================================
print("\n" + "=" * 60)
print("3. УНИКАЛЬНОСТЬ")
print("=" * 60)

unique_prompts = len(set(ex["prompt"] for ex in data))
unique_completions = len(set(ex["completion"] for ex in data))

print(f"  Уникальных prompt:     {unique_prompts:5} ({unique_prompts/len(data)*100:5.1f}%)")
print(f"  Уникальных completion: {unique_completions:5} ({unique_completions/len(data)*100:5.1f}%)")

# Конфликты
prompt_to_completion = {}
for ex in data:
    prompt_to_completion.setdefault(ex["prompt"], set()).add(ex["completion"])
conflicts = sum(1 for v in prompt_to_completion.values() if len(v) > 1)
print(f"  Конфликтов prompt:     {conflicts:5}")

# ============================================================
# 4. ДЛИНА В ТОКЕНАХ (главное!)
# ============================================================
print("\n" + "=" * 60)
print("4. ДЛИНА В ТОКЕНАХ (Qwen tokenizer)")
print("=" * 60)

print("📦 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
print("✅ Токенизатор загружен\n")

import numpy as np

prompt_lens = []
completion_lens = []
total_lens = []

print("Токенизация...")
for i, ex in enumerate(data):
    if i % 500 == 0:
        print(f"  {i}/{len(data)}...")
    p = len(tokenizer.encode(ex["prompt"], add_special_tokens=False))
    c = len(tokenizer.encode(ex["completion"], add_special_tokens=False))
    prompt_lens.append(p)
    completion_lens.append(c)
    total_lens.append(p + c)

def stats(name, arr):
    arr = np.array(arr)
    print(f"\n  {name}:")
    print(f"    min={arr.min():4}  median={int(np.median(arr)):4}  "
          f"p90={int(np.percentile(arr, 90)):4}  p99={int(np.percentile(arr, 99)):4}  "
          f"max={arr.max():4}")

stats("PROMPT", prompt_lens)
stats("COMPLETION", completion_lens)
stats("TOTAL (prompt+completion)", total_lens)

# ============================================================
# 5. ПРОВЕРКА MAX_SEQ_LEN
# ============================================================
print("\n" + "=" * 60)
print(f"5. ПРОВЕРКА MAX_SEQ_LEN={MAX_SEQ_LEN}")
print("=" * 60)

truncated = sum(1 for t in total_lens if t > MAX_SEQ_LEN)
pct_truncated = truncated / len(data) * 100

print(f"\n  Примеров с total > {MAX_SEQ_LEN}: {truncated} ({pct_truncated:.1f}%)")

if truncated > 0:
    print(f"  ⚠️ {truncated} примеров ОБРЕЗАЮТСЯ — completion может потерять ДЕЙСТВИЕ: / СУММА:")
    print(f"  📊 p99 total = {int(np.percentile(total_lens, 99))}")
    print(f"  💡 Рекомендация: MAX_SEQ_LEN >= {int(np.percentile(total_lens, 99)) + 20}")
else:
    print(f"  ✅ Все примеры влезают в {MAX_SEQ_LEN}")

# ============================================================
# 6. ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ
# ============================================================
print("\n" + "=" * 60)
print("6. ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ")
print("=" * 60)

n_train = int(len(data) * 0.9)
effective_batch = BATCH_SIZE * GRAD_ACCUM
steps_per_epoch = n_train // effective_batch
total_steps = steps_per_epoch * NUM_EPOCHS

time_sec = total_steps * GPU_SPEED
time_min = time_sec / 60
time_hours = time_min / 60

print(f"\n  Train примеров:        {n_train}")
print(f"  Batch size:            {BATCH_SIZE}")
print(f"  Grad accumulation:     {GRAD_ACCUM}")
print(f"  Effective batch:       {effective_batch}")
print(f"  Шагов на эпоху:        {steps_per_epoch}")
print(f"  Эпох:                  {NUM_EPOCHS}")
print(f"  Всего шагов:           {total_steps}")
print(f"  Скорость GPU:          {GPU_SPEED} сек/шаг")
print(f"\n  ⏱️ Оценка времени: {time_hours:.1f} часов ({time_min:.0f} минут)")

# ============================================================
# 7. ПРИМЕРЫ
# ============================================================
print("\n" + "=" * 60)
print("7. ПРИМЕРЫ")
print("=" * 60)

for stage in ["preflop", "flop", "turn", "river"]:
    ex = next((e for e in data if e["stage"] == stage), None)
    if ex:
        print(f"\n--- {stage.upper()} ---")
        print(f"PROMPT (первые 300):\n{ex['prompt'][:300]}...")
        print(f"\nCOMPLETION (первые 300):\n{ex['completion'][:300]}...")

# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("  ИТОГ")
print("=" * 60)

ok = True
if pct_truncated > 5:
    print(f"❌ {pct_truncated:.1f}% примеров обрезаются — увеличьте MAX_SEQ_LEN")
    ok = False
else:
    print(f"✅ MAX_SEQ_LEN={MAX_SEQ_LEN} — все влезают")

if actions.get("FOLD", 0) / total_actions < 0.10:
    print(f"❌ FOLD = {actions.get('FOLD', 0)/total_actions*100:.1f}% — мало")
    ok = False
else:
    print(f"✅ FOLD = {actions.get('FOLD', 0)/total_actions*100:.1f}%")

if actions.get("CALL", 0) / total_actions < 0.10:
    print(f"❌ CALL = {actions.get('CALL', 0)/total_actions*100:.1f}% — мало")
    ok = False
else:
    print(f"✅ CALL = {actions.get('CALL', 0)/total_actions*100:.1f}%")

if ok:
    print("\n🎉 ДАТАСЕТ ГОТОВ К ОБУЧЕНИЮ")
else:
    print("\n⚠️ ЕСТЬ ПРОБЛЕМЫ — исправьте перед обучением")









# """
# check_dataset.py - Диагностика датасета. Ищет причину eval_loss=nan.
# Запуск: python check_dataset.py
# """
# import json
# from collections import Counter
# from transformers import AutoTokenizer

# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# # DATASET_FILE = "dataset_combined.json"
# # DATASET_FILE = "./train_2/dataset_unified_test.json"
# DATASET_FILE = "./train_2/dataset_2000_v2.json"
# MAX_SEQ_LEN = 640

# tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# with open(DATASET_FILE, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print("=" * 60)
# print(f"Всего примеров: {len(data)}")
# print("По стадиям:", Counter(x.get("stage", "?") for x in data))

# comps = Counter(x["completion"] for x in data)
# print(f"\nУникальных completion: {len(comps)} ({len(comps)/len(data)*100:.2f}%)")
# print("Топ-10 самых частых:")
# for c, n in comps.most_common(10):
#     print(f"  {n:5d}x  {c.replace(chr(10), ' | ')[:90]}")

# # Сколько токенов занимает prompt и completion
# prompt_lens, comp_lens, broken, no_room = [], [], 0, 0
# for x in data:
#     user_text = tok.apply_chat_template(
#         [{"role": "user", "content": x["prompt"]}],
#         tokenize=False, add_generation_prompt=True,
#     )
#     u_len = len(tok(user_text, add_special_tokens=False)["input_ids"])
#     c_len = len(tok(x["completion"], add_special_tokens=False)["input_ids"]) + 1
#     prompt_lens.append(u_len)
#     comp_lens.append(c_len)
#     if u_len >= MAX_SEQ_LEN:
#         no_room += 1
#     if u_len + c_len > MAX_SEQ_LEN:
#         broken += 1

# prompt_lens.sort()
# def pct(p):
#     return prompt_lens[min(int(len(prompt_lens) * p), len(prompt_lens) - 1)]

# print("\n" + "=" * 60)
# print("ДЛИНА PROMPT (токенов):")
# print(f"  min={prompt_lens[0]}  median={pct(0.5)}  p90={pct(0.9)}  p99={pct(0.99)}  max={prompt_lens[-1]}")
# print(f"  completion: min={min(comp_lens)}  max={max(comp_lens)}")
# print(f"\n❗ Примеров, где completion НЕ ВЛЕЗАЕТ целиком при MAX_SEQ_LEN={MAX_SEQ_LEN}: {broken}")
# print(f"❗ Примеров, где prompt сам по себе >= {MAX_SEQ_LEN} (labels будут ВСЕ -100 → loss=nan): {no_room}")

# # Конфликтующие метки: одинаковый prompt -> разные ответы
# by_prompt = {}
# for x in data:
#     by_prompt.setdefault(x["prompt"], set()).add(x["completion"])
# conflicts = sum(1 for v in by_prompt.values() if len(v) > 1)
# print(f"\n❗ Одинаковых prompt с РАЗНЫМИ ответами: {conflicts}")

# print("\nРЕКОМЕНДАЦИЯ:")
# if no_room or broken:
#     need = pct(0.99) + max(comp_lens) + 8
#     print(f"  Подними MAX_SEQ_LEN до {need} (или сократи prompt).")
#     print("  Иначе eval_loss будет nan.")
# else:
#     print("  По длине всё ок, причина nan в другом (fp16 на Pascal).")