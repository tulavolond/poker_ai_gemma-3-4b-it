# prepare_data.py
import json, re, hashlib
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
raw = open("./dataset_21_11/poker_raw.txt", encoding="utf-8").read()
raw = re.sub(r"//[^\n]*", "", raw)   # безопасно: внутри строк твоего датасета '//' нет
data = json.loads("[" + raw + "]")

seen, uniq = set(), []
for ex in data:
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    key = hashlib.md5(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if key not in seen:
        seen.add(key); uniq.append(ex)

with open("./dataset_21_11/poker.jsonl", "w", encoding="utf-8") as f:
    for ex in uniq:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

tok = AutoTokenizer.from_pretrained(MODEL)   # скачает только токенизатор, ~11 МБ
lens = []
for ex in uniq:
    text = tok.apply_chat_template(ex["messages"], tokenize=False)   # строка, без токенизации
    lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
print(f"примеров: {len(data)} -> {len(uniq)}; токенов: max={max(lens)}, avg={sum(lens)//len(lens)}")