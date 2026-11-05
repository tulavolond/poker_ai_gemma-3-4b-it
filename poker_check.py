"""Проверка обучения: base vs LoRA. Точность + loss на тесте."""

import os
import json

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from poker_common import build_prompt, encode_example, normalize, CLASS_RU

MODEL_ID = "google/gemma-3-4b-it"
LORA_PATH = "./poker_lora"
TEST_FILE = "data_test.jsonl"
LIMIT = 40          # поставь None, чтобы прогнать весь тест
MAX_NEW_TOKENS = 24  # ответ короткий, генерация быстрая

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

with open(TEST_FILE, encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]
if LIMIT:
    examples = examples[:LIMIT]

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Загрузка базовой модели...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
base_model.eval()


@torch.no_grad()
def predict(model, board, hand):
    prompt = build_prompt(tokenizer, board, hand)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


@torch.no_grad()
def mean_loss(model, examples):
    losses = []
    for ex in examples:
        enc = encode_example(tokenizer, ex["board"], ex["hand"], ex["answer"])
        batch = {k: torch.tensor([v], dtype=torch.long).to(model.device) for k, v in enc.items()}
        losses.append(model(**batch).loss.item())
    return sum(losses) / len(losses)


def evaluate(model, name):
    print("\n" + "=" * 70)
    print(f"ОЦЕНКА: {name}")
    print("=" * 70)

    exact = 0
    correct_class = 0
    preds = []

    for i, ex in enumerate(examples, 1):
        pred = predict(model, ex["board"], ex["hand"])
        preds.append(pred)

        if normalize(pred) == normalize(ex["answer"]):
            exact += 1
        if normalize(CLASS_RU[ex["cls"]]) in normalize(pred):
            correct_class += 1

        if i % 10 == 0:
            print(f"  {i}/{len(examples)}")

    n = len(examples)
    stats = {
        "exact_match": round(exact / n, 3),
        "class_accuracy": round(correct_class / n, 3),
        "eval_loss": round(mean_loss(model, examples), 4),
    }

    print(f"\nТочность формата (строка совпала целиком): {stats['exact_match']}")
    print(f"Угадана комбинация:                      {stats['class_accuracy']}")
    print(f"Loss на тесте:                           {stats['eval_loss']}")

    return stats, preds


base_stats, base_preds = evaluate(base_model, "БАЗОВАЯ МОДЕЛЬ (без LoRA)")

print("\nПодключение LoRA-адаптера...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=False)
lora_model.eval()

lora_stats, lora_preds = evaluate(lora_model, "МОДЕЛЬ С LoRA")

print("\n" + "=" * 70)
print("ИТОГ")
print("=" * 70)
print(f"{'метрика':<20}{'base':>12}{'lora':>12}")
for k in base_stats:
    print(f"{k:<20}{base_stats[k]:>12}{lora_stats[k]:>12}")

print("\n" + "=" * 70)
print("ПРИМЕРЫ ОТВЕТОВ")
print("=" * 70)
for ex, b, l in list(zip(examples, base_preds, lora_preds))[:5]:
    print(f"\nстол: {ex['board']} | рука: {ex['hand']}")
    print(f"эталон: {ex['answer']}")
    print(f"base:   {b[:150]}")
    print(f"lora:   {l[:150]}")