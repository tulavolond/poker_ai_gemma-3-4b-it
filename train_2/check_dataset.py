"""
check_dataset.py - Диагностика датасета. Ищет причину eval_loss=nan.
Запуск: python check_dataset.py
"""
import json
from collections import Counter
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# DATASET_FILE = "dataset_combined.json"
# DATASET_FILE = "./train_2/dataset_unified_test.json"
DATASET_FILE = "./train_2/dataset_2000_v2.json"
MAX_SEQ_LEN = 688

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 60)
print(f"Всего примеров: {len(data)}")
print("По стадиям:", Counter(x.get("stage", "?") for x in data))

comps = Counter(x["completion"] for x in data)
print(f"\nУникальных completion: {len(comps)} ({len(comps)/len(data)*100:.2f}%)")
print("Топ-10 самых частых:")
for c, n in comps.most_common(10):
    print(f"  {n:5d}x  {c.replace(chr(10), ' | ')[:180]}")

# Сколько токенов занимает prompt и completion
prompt_lens, comp_lens, broken, no_room = [], [], 0, 0
for x in data:
    user_text = tok.apply_chat_template(
        [{"role": "user", "content": x["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )
    u_len = len(tok(user_text, add_special_tokens=False)["input_ids"])
    c_len = len(tok(x["completion"], add_special_tokens=False)["input_ids"]) + 1
    prompt_lens.append(u_len)
    comp_lens.append(c_len)
    if u_len >= MAX_SEQ_LEN:
        no_room += 1
    if u_len + c_len > MAX_SEQ_LEN:
        broken += 1

prompt_lens.sort()
def pct(p):
    return prompt_lens[min(int(len(prompt_lens) * p), len(prompt_lens) - 1)]

print("\n" + "=" * 60)
print("ДЛИНА PROMPT (токенов):")
print(f"  min={prompt_lens[0]}  median={pct(0.5)}  p90={pct(0.9)}  p99={pct(0.99)}  max={prompt_lens[-1]}")
print(f"  completion: min={min(comp_lens)}  max={max(comp_lens)}")
print(f"\n❗ Примеров, где completion НЕ ВЛЕЗАЕТ целиком при MAX_SEQ_LEN={MAX_SEQ_LEN}: {broken}")
print(f"❗ Примеров, где prompt сам по себе >= {MAX_SEQ_LEN} (labels будут ВСЕ -100 → loss=nan): {no_room}")

# Конфликтующие метки: одинаковый prompt -> разные ответы
by_prompt = {}
for x in data:
    by_prompt.setdefault(x["prompt"], set()).add(x["completion"])
conflicts = sum(1 for v in by_prompt.values() if len(v) > 1)
print(f"\n❗ Одинаковых prompt с РАЗНЫМИ ответами: {conflicts}")

print("\nРЕКОМЕНДАЦИЯ:")
if no_room or broken:
    need = pct(0.99) + max(comp_lens) + 8
    print(f"  Подними MAX_SEQ_LEN до {need} (или сократи prompt).")
    print("  Иначе eval_loss будет nan.")
else:
    print("  По длине всё ок, причина nan в другом (fp16 на Pascal).")