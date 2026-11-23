import json
from pathlib import Path
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MAX_LEN = 1024                      # поставь ТО ЖЕ значение, что в train_lora.py
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
rows = [json.loads(l) for l in open(HERE / "train.jsonl", encoding="utf-8")]
print("rows:", len(rows))

def encode(ex):                     # точная копия логики из train_lora.py
    msgs = ex["messages"]; ids = []; labels = []; prev = ""
    for i, m in enumerate(msgs):
        cur = tok.apply_chat_template(msgs[:i+1], tokenize=False, add_generation_prompt=False)
        seg = tok(cur[len(prev):], add_special_tokens=False)["input_ids"]; prev = cur
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
    return ids[:MAX_LEN], labels[:MAX_LEN]

tot, sup, bad = [], [], 0
for ex in rows:
    ids, lab = encode(ex)
    s = sum(1 for x in lab if x != -100)
    tot.append(len(ids)); sup.append(s); bad += s < 20
print(f"токенов всего:      max={max(tot)} avg={sum(tot)//len(tot)}")
print(f"обучаемых токенов:  min={min(sup)} avg={sum(sup)//len(sup)} max={max(sup)}")
print(f"примеров с <20 обучаемых токенов: {bad}")

ids, lab = encode(rows[0])
print("\n--- что реально размечено для обучения (1-й пример) ---")
print(tok.decode([i for i, l in zip(ids, lab) if l != -100])[:600])
print("\n--- assistant в файле (1-й пример) ---")
print(rows[0]["messages"][2]["content"][:400])

cfg = HERE / "qwen25-stage1-lora" / "adapter_config.json"
if cfg.exists():
    c = json.load(open(cfg, encoding="utf-8"))
    print("\nadapter:", {k: c.get(k) for k in ("r", "lora_alpha", "target_modules")})