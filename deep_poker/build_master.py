# build_master.py
import json
import random
import hashlib
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
random.seed(42)

MASTER_SYSTEM = """Ты — GTO-эксперт по Техасскому Холдему (NLH Cash). Факты от движка (скан, категория, эквити, пот-оддсы, стиль розыгрыша) считаются истинными и не пересчитываются.

ФОРМАТ:
1. Сначала короткий анализ в теге <thought>.
2. Затем один итоговый тег: <action> для решений (Фолд/Колл/Рейз/3-бет) или <definition> для победителя на шоудауне.
3. После закрывающего тега немедленно остановись.

ПРАВИЛА:
- Если дан "Скан:" -> сравни категории и кикеры по скану и выдай <definition>.
- Если даны "эквити" и "пот-оддсы" -> эквити >= пот-оддсы: Колл; иначе Фолд.
- Если даны "Категория" и "Все сбросили (Unopened pot)" -> префлоп открытие банка: Рейз или Фолд по категории и позиции.
- Если даны "Категория" и "Стиль розыгрыша" -> игра против рейза:
  * Вэлью 3-бет -> 3-бет;
  * Позиционный розыгрыш или Сет-майнинг -> Колл;
  * Недостаточно эквити -> Фолд."""

# Если какие-то файлы у тебя называются иначе — поправь пути здесь.
SOURCES = {
    "train": [
        ("stage1", [
            "../dataset_21_11/stage1_v5/train_v61.jsonl",
            "../dataset_21_11/stage1_v5/train_v6.jsonl",
            "../dataset_21_11/stage1_v5/train_v5.jsonl",
            "../dataset_21_11/stage1_v5/train.jsonl",
        ]),
        ("stage2", [
            "../dataset_21_11/stage2/train_bet.jsonl",
        ]),
        ("stage3", [
            "../dataset_21_11/stage3/train_preflop.jsonl",
        ]),
        ("stage4", [
            "../dataset_21_11/stage4/train_facing_v2.jsonl",
            "../dataset_21_11/stage4/train_facing.jsonl",
        ]),
    ],
    "dev": [
        ("stage1", [
            "../dataset_21_11/stage1_v5/dev_v61.jsonl",
            "../dataset_21_11/stage1_v5/dev_v6.jsonl",
            "../dataset_21_11/stage1_v5/dev.jsonl",
        ]),
        ("stage2", [
            "../dataset_21_11/stage2/dev_bet.jsonl",
        ]),
        ("stage3", [
            "../dataset_21_11/stage3/dev_preflop.jsonl",
        ]),
        ("stage4", [
            "../dataset_21_11/stage4/dev_facing_v2.jsonl",
            "../dataset_21_11/stage4/dev_facing.jsonl",
        ]),
    ],
}

# Ограничение для train, чтобы один этап не задавил остальные.
# Для dev не ограничиваем.
MAX_PER_STAGE_TRAIN = 1500


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

            # Подменяем system-промпт на единый master-промпт
            ex["messages"][0]["content"] = MASTER_SYSTEM

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

        # Для train можно ограничить количество примеров на этап
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
    build("train", HERE / "train_master.jsonl")
    build("dev", HERE / "dev_master.jsonl")