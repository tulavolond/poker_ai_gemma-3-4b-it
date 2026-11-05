"""Общие функции: промпт, формат ответа, токенизация. Одни и те же для train и check."""

INSTRUCTION = (
    "Ты — покерный анализатор. Определи итоговую комбинацию из 7 карт на ривере. "
    "Ответь ровно одной строкой в формате: Комбинация: <название>"
)

CLASS_RU = {
    1: "Стрит-флеш",
    2: "Каре",
    3: "Фулл-хаус",
    4: "Флеш",
    5: "Стрит",
    6: "Сет",
    7: "Две пары",
    8: "Пара",
    9: "Старшая карта",
}


def target_text(cls: int) -> str:
    """Эталонный ответ модели."""
    return f"Комбинация: {CLASS_RU[cls]}"


def build_prompt(tokenizer, board: str, hand: str) -> str:
    """Промпт. У Gemma нет роли system, поэтому инструкция внутри user."""
    user_content = f"{INSTRUCTION}\n\nна столе {board}, у меня {hand}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_example(tokenizer, board, hand, answer, max_length=768):
    """Пример для обучения: loss считаем ТОЛЬКО по ответу, промпт замаскирован -100."""
    prompt = build_prompt(tokenizer, board, hand)

    # add_special_tokens=False, чтобы не получить второй BOS (его уже добавил шаблон)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def normalize(text: str) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())