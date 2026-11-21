import json, sys

path = "poker_test_v1.jsonl"
ok, bad = 0, 0

with open(path, encoding="utf-8-sig") as f:  # utf-8-sig переживает BOM от Блокнота
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            print(f"[{i}] ОШИБКА: markdown-ограждение попало в файл"); bad += 1; continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i}] НЕВАЛИДНЫЙ JSON: {e}"); bad += 1; continue

        msgs = obj.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            print(f"[{i}] ОШИБКА: messages должен быть списком из 3 сообщений"); bad += 1; continue
        if [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            print(f"[{i}] ОШИБКА: неверный порядок ролей"); bad += 1; continue

        a = msgs[2]["content"]
        if a.count("<thought>") != 1 or a.count("</thought>") != 1:
            print(f"[{i}] ОШИБКА: сломан тег <thought>"); bad += 1; continue
        if not (("<action>" in a and "</action>" in a) or ("<definition>" in a and "</definition>" in a)):
            print(f"[{i}] ОШИБКА: нет закрывающего <action>/<definition>"); bad += 1; continue
        ok += 1

print(f"\nИтого: валидных {ok}, битых {bad}")