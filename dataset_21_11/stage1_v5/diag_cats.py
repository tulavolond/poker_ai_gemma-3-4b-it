# diag_cats.py — положи в stage1_v5, запускается за секунды
import json
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
NAME = {0:"старшая карта",1:"пара",2:"две пары",3:"сет",4:"стрит",5:"флеш",6:"фулл-хаус",7:"каре",8:"стрит-флеш"}
items = [json.loads(l) for l in open(HERE/"dev.jsonl", encoding="utf-8")]
raws  = [json.loads(l) for l in open(HERE/"answers_dev.jsonl", encoding="utf-8")]
err, tot, nolines = Counter(), Counter(), 0
for ex, r in zip(items, raws):
    if ex["task"] != "compare": continue
    lines = [l for l in r["raw"].splitlines() if l.startswith("И1:") or l.startswith("И2:")]
    if len(lines) != 2: nolines += 1; continue
    for i, l in enumerate(lines):
        claimed = l.rsplit("→", 1)[-1].strip(); true = NAME[ex[f"cat{i+1}"]]
        tot[true] += 1
        if claimed != true: err[f"{true}  →  {claimed}"] += 1
# print(f"категорий проверено {sum(tot.values())}, ошибок {sum(err.values())}, без скана {nolines}")
# for k, v in err.most_common(15): print(f"  {v:3}  {k}")
# # print("\nпо категориям:", {NAME[k]: f"{tot[k]}" for k in sorted(tot)})
# print("\nпо категориям:", {NAME[k]: f"{tot.get(k, 0)}" for k in sorted(NAME)})
# ... (весь код выше остается без изменений) ...

print(f"категорий проверено {sum(tot.values())}, ошибок {sum(err.values())}, без скана {nolines}")

for k, v in err.most_common(15): 
    print(f"  {v:3}  {k}")

# Исправленный вывод: итерируемся по NAME, берем значения из tot
# Используем .get для страховки от отсутствующих ключей
summary = {NAME[k]: f"{tot.get(k, 0)}" for k in sorted(NAME)}
print(f"\nпо категориям: {summary}")