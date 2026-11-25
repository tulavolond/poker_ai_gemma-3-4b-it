# build_decisions_master.py
import json
import random
import hashlib
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
random.seed(42)

# Только решения: Stage 2 + Stage 3 + Stage 4.
# Stage 1 (шоудаун) пока не берём, чтобы не портить формат <action>.
SOURCES = {
    "train": [
        ("stage2", [
            "stage2/train_bet.jsonl",
        ]),
        ("stage3", [
            "stage3/train_preflop.jsonl",
        ]),
        ("stage4", [
            "stage4/train_facing_v2.jsonl",
            "stage4/train_facing.jsonl",
        ]),
    ],
    "dev": [
        ("stage2", [
            "stage2/dev_bet.jsonl",
        ]),
        ("stage3", [
            "stage3/dev_preflop.jsonl",
        ]),
        ("stage4", [
            "stage4/dev_facing_v2.jsonl",
            "stage4/dev_facing.jsonl",
        ]),
    ],
}

# Чтобы один этап не задавил другие.
# У тебя этапы примерно по 400–900 примеров, так что 1200 безопасно.
MAX_PER_STAGE_TRAIN = 1200


def load_rows(path: Path, keep_meta: bool):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            if "messages" not in ex or len(ex["messages"]) != 3:
                continue

            # ВАЖНО:
            # НЕ заменяем system-промпт.
            # Оставляем оригинальный system из stage2/stage3/stage4.
            if keep_meta:
                rows.append(ex)
            else:
                rows.append({"messages": ex["messages"]})

    return rows


def build(split: str, out_name: str):
    seen = set()
    all_rows = []
    counts = Counter()

    for stage, candidates in SOURCES[split]:
        path = None

        for cand in candidates:
            p = HERE / cand
            if p.exists():
                path = p
                break

        if path is None:
            print(f"⚠️ {split}/{stage}: файл не найден. Кандидаты: {candidates}")
            continue

        rows = load_rows(path, keep_meta=(split != "train"))

        # Дедупликация по вопросу и ответу
        uniq = []
        for ex in rows:
            user = ex["messages"][1]["content"]
            assistant = ex["messages"][2]["content"]
            key = hashlib.md5((user + "\n" + assistant).encode("utf-8")).hexdigest()

            if key in seen:
                continue

            seen.add(key)
            uniq.append(ex)

        if split == "train" and len(uniq) > MAX_PER_STAGE_TRAIN:
            random.shuffle(uniq)
            uniq = uniq[:MAX_PER_STAGE_TRAIN]

        counts[stage] = len(uniq)
        all_rows.extend(uniq)

        print(f"✅ {split}/{stage}: {len(uniq)} <- {path}")

    random.shuffle(all_rows)

    out_path = HERE / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"🚀 {split}: {len(all_rows)} -> {out_path}")
    print(f"   распределение: {dict(counts)}")
    print()


if __name__ == "__main__":
    build("train", HERE / "train_decisions.jsonl")
    build("dev", HERE / "dev_decisions.jsonl")