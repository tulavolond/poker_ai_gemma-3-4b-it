# sync_system.py — положи в stage1_v5
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
train_sys = json.loads(open(HERE / "train.jsonl", encoding="utf-8").readline())["messages"][0]["content"]

for fname in ("dev.jsonl", "test.jsonl"):
    p = HERE / fname
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    differed = rows[0]["messages"][0]["content"] != train_sys
    for ex in rows:
        ex["messages"][0]["content"] = train_sys      # меняем ТОЛЬКО system
    with open(p, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{fname}: {len(rows)} строк | system отличался от train: {differed} → синхронизирован")